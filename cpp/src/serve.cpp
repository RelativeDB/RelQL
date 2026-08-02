/* serve.cpp — rt_serve: the model-serving web backend.
 *
 * Context creation lives entirely in the Python package; what arrives here is
 * the prepared token batch (text still as raw strings) plus scoring metadata
 * — never RelQL text, rows, or anything schema-shaped that would need
 * parsing. This process owns the two models: the RT-J transformer and the
 * pinned MiniLM text encoder (all text embedding happens here), so weights,
 * the encoder, and the GPU sit in one place while any number of query
 * processes stay light.
 *
 * Protocol (JSON over HTTP/1.1; relativedb.remote.RemoteScorer is the
 * reference client):
 *   GET  /health                -> {"status":"ok","device":…}
 *                               -> {"embeddings":[[384]…]}
 *   POST /v1/forward            {"model_uri":…, "output":…, "batch":{…},
 *                                "query"?:…, "task_type"?:…, "session"?:…}
 *                               -> {"scores":…} / {"features":…}
 *                                  (+ "target_text" for
 *                                   target_scores_and_text)
 *   POST /v1/flat_features      {"spec":{…},"contexts":[…]}
 *                               -> {"features":[[…]…]}  (tree backend)
 *
 * The batch encoding matches relativedb.remote.encode_batch: integer
 * channels as nested lists, number/datetime channels already bfloat16-
 * rounded, col_phrases indexed by col id, texts deduplicated with per-token
 * text_idx. The two 384-d channels are materialized HERE from the native
 * MiniLM encoder and bf16-rounded, mirroring NativeScorer._materialize.
 *
 * HTTP is served by libh2o (evloop flavor): one event-loop thread owns
 * accept/parse/respond and never blocks; POST bodies are handed to a worker
 * pool (RT_SERVE_WORKERS) that decodes, embeds, and waits on the GPU worker,
 * then posts the finished JSON back to the loop via an h2o multithread
 * receiver. The wire is JSON floats — every bf16 value round-trips exactly
 * through "%.9g".
 *
 * Forwards are dynamically batched: connection threads decode and embed,
 * then queue the materialized rt::Batch for a single GPU worker. The worker
 * drains everything queued for the same (model, head flags) — padding rows
 * to a common S — and runs ONE rt::forward for the lot, so concurrent small
 * requests coalesce instead of serializing on the model. Sequences never
 * interact inside the model (sorting, attention groups, and reductions in
 * rt.cpp are all per batch row), so coalescing cannot leak state between
 * requests; it only changes GPU launch shapes. A lone request dispatches
 * immediately. Caps: RT_SERVE_MAX_BATCH_ROWS (64) and
 * RT_SERVE_MAX_BATCH_TOKENS (262144); RT_SERVE_BATCH_WAIT_MS (0) optionally
 * lingers after the first request so a burst can accumulate.
 */
#include <arpa/inet.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#include <h2o.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "flat.hpp"
#include "json.hpp"
#include "minilm.hpp"
#include "rt.hpp"
#include "rt_internal.hpp"

namespace {

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

float bf16_round(float v) {
  uint32_t bits;
  std::memcpy(&bits, &v, 4);
  bits = (bits + 0x7FFFu + ((bits >> 16) & 1u)) & 0xFFFF0000u;
  float out;
  std::memcpy(&out, &bits, 4);
  return out;
}

void append_float(std::string& out, float v) {
  char buf[32];
  if (std::isfinite(v)) {
    std::snprintf(buf, sizeof(buf), "%.9g", (double)v);
  } else if (std::isnan(v)) {
    std::snprintf(buf, sizeof(buf), "null");   // JSON has no NaN
  } else {
    std::snprintf(buf, sizeof(buf), v > 0 ? "1e999" : "-1e999");
  }
  out += buf;
}

void append_float_array(std::string& out, const float* v, size_t n) {
  out += "[";
  for (size_t i = 0; i < n; ++i) {
    if (i) out += ",";
    append_float(out, v[i]);
  }
  out += "]";
}

std::string json_escape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 2);
  for (unsigned char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out.push_back((char)c);
        }
    }
  }
  return out;
}

// hf://org/repo/sub -> HF-cache snapshot path; plain paths pass through.
// The server never downloads: a cold cache is an operator problem, and the
// error says exactly what to fetch.
// A checkpoint directory holds model.safetensors or a single-dtype variant
// (model.f16.safetensors — e.g. RelativeDB/rt-j-fp16). fp32 wins when both.
const char* const kModelNames[] = {"model.safetensors",
                                   "model.f16.safetensors"};

std::string resolve_model_uri(const std::string& uri) {
  if (uri.rfind("hf://", 0) != 0) {
    if (fs::is_directory(uri)) {
      for (const char* name : kModelNames) {
        const fs::path direct = fs::path(uri) / name;
        if (fs::exists(direct)) return direct.string();
      }
      throw std::runtime_error("directory " + uri + " has no model.safetensors");
    }
    return uri;
  }
  std::string rest = uri.substr(5);
  while (!rest.empty() && rest.front() == '/') rest.erase(0, 1);
  const size_t s1 = rest.find('/');
  if (s1 == std::string::npos)
    throw std::runtime_error("malformed hf:// URI: " + uri);
  const size_t s2 = rest.find('/', s1 + 1);
  const std::string repo = rest.substr(0, s2);   // org/repo
  const std::string sub = s2 == std::string::npos ? "" : rest.substr(s2 + 1);
  std::string cache_repo = "models--" + repo;
  for (char& c : cache_repo)
    if (c == '/') c = '-', cache_repo.insert(&c - cache_repo.data() + 1, "-");
  // (org/repo -> models--org--repo)
  std::string home;
  if (const char* hf = std::getenv("HF_HOME")) home = std::string(hf) + "/hub";
  else if (const char* h = std::getenv("HOME"))
    home = std::string(h) + "/.cache/huggingface/hub";
  const fs::path snaps = fs::path(home) / cache_repo / "snapshots";
  std::error_code ec;
  fs::path best;
  fs::file_time_type best_time{};
  for (const auto& entry : fs::directory_iterator(snaps, ec)) {
    fs::path base = entry.path();
    if (!sub.empty()) base /= sub;
    fs::path cand;
    for (const char* name : kModelNames)
      if (fs::exists(base / name, ec)) {
        cand = base / name;
        break;
      }
    if (cand.empty()) continue;
    const auto t = fs::last_write_time(entry.path(), ec);
    if (best.empty() || t > best_time) { best = cand; best_time = t; }
  }
  if (best.empty())
    throw std::runtime_error(
        uri + " is not in the HF cache (" + snaps.string() +
        "); download the checkpoint before serving");
  return best.string();
}

// ---------------------------------------------------------------------------
// service state
// ---------------------------------------------------------------------------

int env_int(const char* name, int dflt) {
  const char* v = std::getenv(name);
  return v ? std::atoi(v) : dflt;
}

// One /v1/forward waiting for the GPU worker: a fully materialized batch
// (text channels already embedded and scattered) plus where to deliver.
struct PendingFwd {
  const rt::Model* model = nullptr;
  std::string group;   // model path + head flags: coalesce only within this
  rt::Batch b;
  rt::ForwardOpts opts;
  std::promise<rt::Output> prom;
};

struct Service {
  std::unique_ptr<minilm::MiniLM> encoder;
  rt::Device device = rt::Device::CPU;
  int n_threads = 0;

  std::mutex models_mu;
  std::unordered_map<std::string, std::unique_ptr<rt::Model>> models;

  std::mutex embed_mu;
  std::unordered_map<std::string, std::vector<float>> embed_cache;   // raw

  std::mutex q_mu;
  std::condition_variable q_cv;
  std::deque<std::unique_ptr<PendingFwd>> fwd_q;
  const int max_batch_rows = env_int("RT_SERVE_MAX_BATCH_ROWS", 64);
  const long max_batch_tokens = env_int("RT_SERVE_MAX_BATCH_TOKENS", 262144);
  const int batch_wait_ms = env_int("RT_SERVE_BATCH_WAIT_MS", 0);

  std::atomic<uint64_t> forwards{0}, embeds{0};
  std::atomic<uint64_t> gpu_batches{0}, coalesced{0};

  const rt::Model& model_for(const std::string& uri) {
    const std::string path = resolve_model_uri(uri);
    std::lock_guard<std::mutex> lock(models_mu);
    auto it = models.find(path);
    if (it == models.end()) {
      std::fprintf(stderr, "[rt_serve] loading %s\n", path.c_str());
      auto m = std::make_unique<rt::Model>(rt::Model::load(path));
      it = models.emplace(path, std::move(m)).first;
    }
    return *it->second;
  }

  // Raw (un-normalized) embeddings with a process-wide cache: schema phrases
  // repeat on every forward, so the encoder mostly runs on novel cell text.
  void embed_raw(const std::vector<std::string>& texts, float* out) {
    std::vector<size_t> missing;
    {
      std::lock_guard<std::mutex> lock(embed_mu);
      for (size_t i = 0; i < texts.size(); ++i) {
        auto it = embed_cache.find(texts[i]);
        if (it != embed_cache.end())
          std::memcpy(out + i * minilm::kDim, it->second.data(),
                      minilm::kDim * sizeof(float));
        else
          missing.push_back(i);
      }
    }
    if (missing.empty()) return;
    std::vector<std::string> todo;
    todo.reserve(missing.size());
    for (size_t i : missing) todo.push_back(texts[i]);
    std::vector<float> fresh(todo.size() * minilm::kDim);
    std::string err;
    if (!encoder->encode(todo, /*normalize=*/false, fresh.data(), &err))
      throw std::runtime_error("text encoder failed: " + err);
    embeds += todo.size();
    std::lock_guard<std::mutex> lock(embed_mu);
    for (size_t k = 0; k < missing.size(); ++k) {
      const float* v = fresh.data() + k * minilm::kDim;
      std::memcpy(out + missing[k] * minilm::kDim, v,
                  minilm::kDim * sizeof(float));
      embed_cache.emplace(todo[k], std::vector<float>(v, v + minilm::kDim));
    }
  }

  // Queue a materialized batch for the GPU worker and wait for its slice of
  // the coalesced output. The returned Output's S may exceed b.S (rows are
  // padded to the widest request in the GPU batch).
  rt::Output forward_queued(const rt::Model& model, const std::string& path,
                            rt::Batch&& b, const rt::ForwardOpts& opts) {
    auto p = std::make_unique<PendingFwd>();
    p->model = &model;
    p->group = path;
    p->group += opts.want_text_head ? "|T" : "";
    p->group += opts.want_target_features ? "|F" : "";
    p->b = std::move(b);
    p->opts = opts;
    std::future<rt::Output> fut = p->prom.get_future();
    {
      std::lock_guard<std::mutex> lock(q_mu);
      fwd_q.push_back(std::move(p));
    }
    q_cv.notify_one();
    return fut.get();
  }
};

// ---------------------------------------------------------------------------
// GPU worker: drain compatible pending forwards into one rt::forward
// ---------------------------------------------------------------------------

// items -> one batch, each row right-padded to S. Padding channel values
// mirror relativedb.scoring.SequenceBackend._collate: is_padding=1, f2p=-1,
// everything else zero.
rt::Batch merge_batches(const std::vector<PendingFwd*>& items, int S) {
  rt::Batch m;
  m.S = S;
  m.B = 0;
  for (const PendingFwd* p : items) m.B += p->b.B;
  const size_t BS = (size_t)m.B * S;
  m.node_idxs.assign(BS, 0);
  m.f2p.assign(BS * rt::kMaxF2p, -1);
  m.col_idxs.assign(BS, 0);
  m.table_idxs.assign(BS, 0);
  m.is_padding.assign(BS, 1);
  m.sem_types.assign(BS, 0);
  m.is_target.assign(BS, 0);
  m.number_v.assign(BS, 0.0f);
  m.datetime_v.assign(BS, 0.0f);
  m.boolean_v.assign(BS, 0.0f);
  m.text_v.assign(BS * rt::kDText, 0.0f);
  m.col_name_v.assign(BS * rt::kDText, 0.0f);
  int r0 = 0;
  for (const PendingFwd* p : items) {
    const rt::Batch& b = p->b;
    for (int r = 0; r < b.B; ++r) {
      const size_t src = (size_t)r * b.S, dst = (size_t)(r0 + r) * S;
      std::memcpy(&m.node_idxs[dst], &b.node_idxs[src], b.S * 8);
      std::memcpy(&m.f2p[dst * rt::kMaxF2p], &b.f2p[src * rt::kMaxF2p],
                  (size_t)b.S * rt::kMaxF2p * 8);
      std::memcpy(&m.col_idxs[dst], &b.col_idxs[src], b.S * 8);
      std::memcpy(&m.table_idxs[dst], &b.table_idxs[src], b.S * 8);
      std::memcpy(&m.is_padding[dst], &b.is_padding[src], b.S);
      std::memcpy(&m.sem_types[dst], &b.sem_types[src], b.S * 8);
      std::memcpy(&m.is_target[dst], &b.is_target[src], b.S);
      std::memcpy(&m.number_v[dst], &b.number_v[src], b.S * 4);
      std::memcpy(&m.datetime_v[dst], &b.datetime_v[src], b.S * 4);
      std::memcpy(&m.boolean_v[dst], &b.boolean_v[src], b.S * 4);
      std::memcpy(&m.text_v[dst * rt::kDText], &b.text_v[src * rt::kDText],
                  (size_t)b.S * rt::kDText * 4);
      std::memcpy(&m.col_name_v[dst * rt::kDText],
                  &b.col_name_v[src * rt::kDText],
                  (size_t)b.S * rt::kDText * 4);
    }
    r0 += b.B;
  }
  return m;
}

rt::Output slice_output(const rt::Output& out, int r0, int B) {
  rt::Output o;
  o.B = B;
  o.S = out.S;
  const size_t lo = (size_t)r0 * out.S, hi = (size_t)(r0 + B) * out.S;
  o.sort_idxs.assign(out.sort_idxs.begin() + lo, out.sort_idxs.begin() + hi);
  o.sorted_is_target.assign(out.sorted_is_target.begin() + lo,
                            out.sorted_is_target.begin() + hi);
  o.yhat_number.assign(out.yhat_number.begin() + lo,
                       out.yhat_number.begin() + hi);
  if (!out.yhat_text.empty())
    o.yhat_text.assign(out.yhat_text.begin() + lo * rt::kDText,
                       out.yhat_text.begin() + hi * rt::kDText);
  if (!out.target_features.empty())
    o.target_features.assign(
        out.target_features.begin() + (size_t)r0 * rt::kDModel,
        out.target_features.begin() + (size_t)(r0 + B) * rt::kDModel);
  return o;
}

// A coalesced group that finished its CPU stage and is waiting for a device.
// `prepare` (sort, attention grouping, embeddings scatter) costs as much CPU
// as the blocks cost GPU, so the two run as a pipeline: prep threads keep
// the bounded queue fed while the single blocks thread owns the device.
struct PreparedGroup {
  std::vector<std::unique_ptr<PendingFwd>> items;
  rt::Batch merged;             // owner of the batch when items were merged
  rt::Output out;
  rt::detail::Prepared prep;
};

std::mutex g_prep_mu;
std::condition_variable g_prep_cv, g_prep_space_cv;
std::deque<std::unique_ptr<PreparedGroup>> g_prepared_q;
constexpr size_t kPreparedDepth = 4;

void fail_group(std::vector<std::unique_ptr<PendingFwd>>& items) {
  for (auto& p : items) {
    try {
      p->prom.set_exception(std::current_exception());
    } catch (const std::future_error&) {   // already satisfied
    }
  }
}

void prep_loop(Service& svc) {
  for (;;) {
    std::vector<std::unique_ptr<PendingFwd>> group;
    {
      std::unique_lock<std::mutex> lk(svc.q_mu);
      svc.q_cv.wait(lk, [&] { return !svc.fwd_q.empty(); });
      if (svc.batch_wait_ms > 0) {
        // optional linger so a burst can accumulate before we drain
        lk.unlock();
        std::this_thread::sleep_for(
            std::chrono::milliseconds(svc.batch_wait_ms));
        lk.lock();
        if (svc.fwd_q.empty()) continue;
      }
      group.push_back(std::move(svc.fwd_q.front()));
      svc.fwd_q.pop_front();
      long rows = group[0]->b.B;
      int S = group[0]->b.S;
      for (auto it = svc.fwd_q.begin(); it != svc.fwd_q.end();) {
        PendingFwd& p = **it;
        const int wide = std::max(S, p.b.S);
        if (p.group == group[0]->group &&
            rows + p.b.B <= svc.max_batch_rows &&
            (rows + p.b.B) * (long)wide <= svc.max_batch_tokens) {
          rows += p.b.B;
          S = wide;
          group.push_back(std::move(*it));
          it = svc.fwd_q.erase(it);
        } else {
          ++it;
        }
      }
    }
    auto pg = std::make_unique<PreparedGroup>();
    try {
      const rt::Batch* b;
      if (group.size() == 1) {
        b = &group[0]->b;
      } else {
        int S = 0;
        std::vector<PendingFwd*> raw;
        raw.reserve(group.size());
        for (auto& p : group) {
          S = std::max(S, p->b.S);
          raw.push_back(p.get());
        }
        pg->merged = merge_batches(raw, S);
        b = &pg->merged;
      }
      const bool host_embed = group[0]->opts.device != rt::Device::CUDA;
      pg->prep = rt::detail::prepare(*group[0]->model, *b, pg->out,
                                     /*debug_taps=*/false, host_embed);
    } catch (...) {
      fail_group(group);
      continue;
    }
    pg->items = std::move(group);
    {
      std::unique_lock<std::mutex> lk(g_prep_mu);
      g_prep_space_cv.wait(lk,
                           [] { return g_prepared_q.size() < kPreparedDepth; });
      g_prepared_q.push_back(std::move(pg));
    }
    g_prep_cv.notify_one();
  }
}

void blocks_loop(Service& svc) {
  for (;;) {
    std::unique_ptr<PreparedGroup> pg;
    {
      std::unique_lock<std::mutex> lk(g_prep_mu);
      g_prep_cv.wait(lk, [] { return !g_prepared_q.empty(); });
      pg = std::move(g_prepared_q.front());
      g_prepared_q.pop_front();
    }
    g_prep_space_cv.notify_one();
    try {
      const rt::Model& model = *pg->items[0]->model;
      const rt::ForwardOpts& opts = pg->items[0]->opts;
      switch (opts.device) {
        case rt::Device::CPU:
          rt::detail::run_blocks_cpu(model, pg->prep, pg->out, opts.n_threads,
                                     false, opts.want_text_head,
                                     opts.want_target_features);
          break;
#ifdef RT_METAL
        case rt::Device::MPS:
          rt::detail::run_blocks_metal(model, pg->prep, pg->out, false,
                                       opts.want_target_features);
          break;
#endif
#ifdef RT_CUDA
        case rt::Device::CUDA:
          rt::detail::run_blocks_cuda(model, pg->prep, pg->out, false,
                                      opts.want_target_features);
          break;
#endif
        default:
          throw std::runtime_error("rt_serve: device backend not compiled in");
      }
      ++svc.gpu_batches;
      if (pg->items.size() == 1) {
        pg->items[0]->prom.set_value(std::move(pg->out));
      } else {
        svc.coalesced += pg->items.size();
        int r0 = 0;
        for (auto& p : pg->items) {
          p->prom.set_value(slice_output(pg->out, r0, p->b.B));
          r0 += p->b.B;
        }
      }
    } catch (...) {
      fail_group(pg->items);
    }
  }
}

// ---------------------------------------------------------------------------
// request handlers
// ---------------------------------------------------------------------------

const relql::JsonValue& need(const relql::JsonValue& obj, const char* key) {
  const relql::JsonValue* v = obj.find(key);
  if (!v) throw std::runtime_error(std::string("missing field '") + key + "'");
  return *v;
}

void fill_i64(const relql::JsonValue& v, std::vector<int64_t>& out) {
  for (const relql::JsonValue& row : v.arr)
    if (row.kind == relql::JsonValue::Kind::Arr)
      for (const relql::JsonValue& x : row.arr) {
        if (x.kind == relql::JsonValue::Kind::Arr)   // f2p [B][S][5]
          for (const relql::JsonValue& y : x.arr) out.push_back((int64_t)y.num);
        else
          out.push_back((int64_t)x.num);
      }
}

void fill_u8(const relql::JsonValue& v, std::vector<uint8_t>& out) {
  for (const relql::JsonValue& row : v.arr)
    for (const relql::JsonValue& x : row.arr) out.push_back((uint8_t)x.num);
}

void fill_f32(const relql::JsonValue& v, std::vector<float>& out) {
  for (const relql::JsonValue& row : v.arr)
    for (const relql::JsonValue& x : row.arr) out.push_back((float)x.num);
}

// One decoded /v?/forward request, wire-format independent: numeric channels
// filled, text still raw strings (phrases first, then texts, in to_embed).
struct DecodedFwd {
  std::string uri, output;
  rt::Batch b;
  std::vector<std::string> to_embed;
  size_t n_phrases = 0;
  std::vector<int64_t> tidx;
};

DecodedFwd decode_forward_json(const relql::JsonValue& req) {
  DecodedFwd d;
  d.uri = need(req, "model_uri").str;
  const relql::JsonValue* outv = req.find("output");
  d.output = outv && outv->kind == relql::JsonValue::Kind::Str
                 ? outv->str
                 : "target_scores";
  const relql::JsonValue& jb = need(req, "batch");
  rt::Batch& b = d.b;
  b.B = (int)need(jb, "b").num;
  b.S = (int)need(jb, "s").num;
  if (b.B <= 0 || b.S <= 0) throw std::runtime_error("batch b/s must be positive");
  const size_t BS = (size_t)b.B * b.S;
  fill_i64(need(jb, "node_idxs"), b.node_idxs);
  fill_i64(need(jb, "f2p"), b.f2p);
  fill_i64(need(jb, "col_idxs"), b.col_idxs);
  fill_i64(need(jb, "table_idxs"), b.table_idxs);
  fill_u8(need(jb, "is_padding"), b.is_padding);
  fill_i64(need(jb, "sem_types"), b.sem_types);
  fill_u8(need(jb, "is_target"), b.is_target);
  fill_f32(need(jb, "number_v"), b.number_v);
  fill_f32(need(jb, "datetime_v"), b.datetime_v);
  if (b.node_idxs.size() != BS || b.f2p.size() != BS * rt::kMaxF2p ||
      b.col_idxs.size() != BS || b.table_idxs.size() != BS ||
      b.is_padding.size() != BS || b.sem_types.size() != BS ||
      b.is_target.size() != BS || b.number_v.size() != BS ||
      b.datetime_v.size() != BS)
    throw std::runtime_error("batch channel shapes disagree with b/s");
  const relql::JsonValue& phrases = need(jb, "col_phrases");
  const relql::JsonValue& texts = need(jb, "texts");
  d.to_embed.reserve(phrases.arr.size() + texts.arr.size());
  for (const relql::JsonValue& p : phrases.arr) d.to_embed.push_back(p.str);
  for (const relql::JsonValue& t : texts.arr) d.to_embed.push_back(t.str);
  d.n_phrases = phrases.arr.size();
  const relql::JsonValue* text_idx = jb.find("text_idx");
  if (text_idx && text_idx->kind == relql::JsonValue::Kind::Arr)
    fill_i64(*text_idx, d.tidx);
  return d;
}

// Binary wire (relativedb.remote encode_batch_bin is the reference client):
//   u32le header_len | header JSON | raw channel arrays, fixed order
// Header: {"model_uri","output","b","s","col_phrases":[…],"texts":[…],
//          "has_text_idx":bool}. Arrays (little-endian, C order):
//   node i32[B,S] | f2p i32[B,S,5] | col i32[B,S] | tab i32[B,S] |
//   pad u8[B,S]  | sem u8[B,S]    | tgt u8[B,S]  | num f32[B,S] |
//   dt  f32[B,S] | tidx i32[B,S] (only when has_text_idx)
// Index channels ride as i32: node/f2p values are context-local node ids and
// col/tab/sem are tiny vocabularies — they widen to the Batch's i64 here.

std::string handle_forward(Service& svc, DecodedFwd&& dec, bool binary) {
  const std::string& uri = dec.uri;
  const std::string& output = dec.output;
  rt::Batch& b = dec.b;
  const size_t BS = (size_t)b.B * b.S;
  b.boolean_v.assign(BS, 0.0f);   // bool_as_num: booleans ride number_v

  // ---- embed schema phrases + text cells, scatter, bf16-round ------------
  std::vector<std::string>& to_embed = dec.to_embed;
  std::vector<float> emb(to_embed.size() * minilm::kDim);
  const uint64_t embeds_before = svc.embeds.load();
  const auto t_embed = std::chrono::steady_clock::now();
  if (!to_embed.empty()) svc.embed_raw(to_embed, emb.data());
  const auto t_scatter = std::chrono::steady_clock::now();
  const size_t n_phrases = dec.n_phrases;
  const size_t n_texts = to_embed.size() - n_phrases;

  b.text_v.assign(BS * rt::kDText, 0.0f);
  b.col_name_v.assign(BS * rt::kDText, 0.0f);
  const std::vector<int64_t>& tidx = dec.tidx;
  for (size_t i = 0; i < BS; ++i) {
    if (b.is_padding[i]) continue;
    const int64_t col = b.col_idxs[i];
    if (col >= 0 && (size_t)col < n_phrases) {
      const float* src = emb.data() + (size_t)col * minilm::kDim;
      float* dst = b.col_name_v.data() + i * rt::kDText;
      for (int d = 0; d < rt::kDText; ++d) dst[d] = bf16_round(src[d]);
    }
    if (!tidx.empty() && tidx[i] >= 0 && (size_t)tidx[i] < n_texts) {
      const float* src =
          emb.data() + (n_phrases + (size_t)tidx[i]) * minilm::kDim;
      float* dst = b.text_v.data() + i * rt::kDText;
      for (int d = 0; d < rt::kDText; ++d) dst[d] = bf16_round(src[d]);
    }
  }

  // ---- forward (via the GPU worker; the batch may ride with others) -------
  const rt::Model& model = svc.model_for(uri);
  rt::ForwardOpts opts;
  opts.device = svc.device;
  opts.n_threads = svc.n_threads;
  opts.debug_taps = false;
  if (output == "target_scores_and_text") {
    opts.want_text_head = true;
    opts.device = rt::Device::CPU;   // text head is CPU-only (rt.hpp)
  }
  if (output == "target_features") opts.want_target_features = true;
  const int B = b.B, S_req = b.S;
  const auto t_fwd = std::chrono::steady_clock::now();
  const rt::Output out =
      svc.forward_queued(model, resolve_model_uri(uri), std::move(b), opts);
  ++svc.forwards;
  const int S_out = out.S;   // >= S_req when coalesced with wider requests
  {
    const auto now = std::chrono::steady_clock::now();
    auto ms = [](auto a, auto b) {
      return std::chrono::duration<double, std::milli>(b - a).count();
    };
    std::fprintf(stderr,
                 "[serve] fwd b=%d s=%d embed=%.0fms(%llu new) "
                 "queue+fwd=%.0fms\n",
                 B, S_req, ms(t_embed, t_scatter),
                 (unsigned long long)(svc.embeds.load() - embeds_before),
                 ms(t_fwd, now));
  }

  // ---- payloads (wire-format independent) ---------------------------------
  std::vector<float> scores, target_text;
  if (output == "target_features") {
    scores.assign(out.target_features.begin(),
                  out.target_features.begin() + (size_t)B * rt::kDModel);
  } else if (output == "token_scores") {
    // yhat_number is in sorted token order; sort_idxs maps back to pre-sort.
    // Coalescing pads rows to S_out; pre-sort indices >= S_req are padding.
    scores.assign(BS, 0.0f);
    for (int r = 0; r < B; ++r)
      for (int s = 0; s < S_out; ++s) {
        const size_t i = (size_t)r * S_out + s;
        if (out.sort_idxs[i] < S_req)
          scores[(size_t)r * S_req + out.sort_idxs[i]] = out.yhat_number[i];
      }
  } else {
    scores.assign(B, 0.0f);
    for (int r = 0; r < B; ++r)
      for (int s = 0; s < S_out; ++s) {
        const size_t i = (size_t)r * S_out + s;
        if (out.sorted_is_target[i]) scores[r] += out.yhat_number[i];
      }
    if (output == "target_scores_and_text") {
      target_text.assign((size_t)B * rt::kDText, 0.0f);
      for (int r = 0; r < B; ++r)
        for (int s = 0; s < S_out; ++s) {
          const size_t i = (size_t)r * S_out + s;
          if (!out.sorted_is_target[i]) continue;
          const float* row = out.yhat_text.data() + i * rt::kDText;
          float* acc = target_text.data() + (size_t)r * rt::kDText;
          for (int d = 0; d < rt::kDText; ++d) acc[d] += row[d];
        }
    }
  }

  if (binary) {
    // u32le header_len | header JSON | scores f32 raw | target_text f32 raw
    std::string hdr = "{\"b\":" + std::to_string(B) +
                      ",\"s\":" + std::to_string(S_req) + ",\"output\":\"" +
                      output + "\",\"device\":\"";
    hdr += rt::device_name(opts.device);
    hdr += "\"}";
    std::string body(4, '\0');
    const uint32_t hlen = (uint32_t)hdr.size();
    std::memcpy(body.data(), &hlen, 4);
    body += hdr;
    body.append(reinterpret_cast<const char*>(scores.data()),
                scores.size() * 4);
    if (!target_text.empty())
      body.append(reinterpret_cast<const char*>(target_text.data()),
                  target_text.size() * 4);
    return body;
  }

  std::string body = "{";
  const char* key = output == "target_features" ? "\"features\":[" : "\"scores\":[";
  body += key;
  if (output == "target_features") {
    for (int r = 0; r < B; ++r) {
      if (r) body += ",";
      append_float_array(body, scores.data() + (size_t)r * rt::kDModel,
                         rt::kDModel);
    }
  } else if (output == "token_scores") {
    for (int r = 0; r < B; ++r) {
      if (r) body += ",";
      append_float_array(body, scores.data() + (size_t)r * S_req, S_req);
    }
  } else {
    for (int r = 0; r < B; ++r) {
      if (r) body += ",";
      append_float(body, scores[r]);
    }
  }
  body += "]";
  if (!target_text.empty()) {
    body += ",\"target_text\":[";
    for (int r = 0; r < B; ++r) {
      if (r) body += ",";
      append_float_array(body, target_text.data() + (size_t)r * rt::kDText,
                         rt::kDText);
    }
    body += "]";
  }
  body += ",\"device\":\"";
  body += rt::device_name(opts.device);
  body += "\"}";
  return body;
}

std::string handle_embed(Service& svc, const relql::JsonValue& req) {
  const relql::JsonValue& texts = need(req, "texts");
  const relql::JsonValue* nv = req.find("normalize");
  const bool normalize = nv && nv->kind == relql::JsonValue::Kind::Bool && nv->b;
  std::vector<std::string> in;
  in.reserve(texts.arr.size());
  for (const relql::JsonValue& t : texts.arr) in.push_back(t.str);
  std::vector<float> out(in.size() * minilm::kDim);
  if (!in.empty()) {
    svc.embed_raw(in, out.data());   // cache holds raw; normalize below
    if (normalize) {
      for (size_t i = 0; i < in.size(); ++i) {
        float* v = out.data() + i * minilm::kDim;
        float n = 0.f;
        for (int d = 0; d < minilm::kDim; ++d) n += v[d] * v[d];
        n = std::max(std::sqrt(n), 1e-12f);
        for (int d = 0; d < minilm::kDim; ++d) v[d] /= n;
      }
    }
  }
  std::string body = "{\"embeddings\":[";
  for (size_t i = 0; i < in.size(); ++i) {
    if (i) body += ",";
    append_float_array(body, out.data() + i * minilm::kDim, minilm::kDim);
  }
  body += "]}";
  return body;
}

std::string handle_flat(const relql::JsonValue& req) {
  const relql::JsonValue& spec = need(req, "spec");
  const relql::JsonValue& contexts = need(req, "contexts");
  const size_t k = relql::flat_spec_size(spec);
  std::string body = "{\"features\":[";
  std::vector<float> row(k);
  for (size_t i = 0; i < contexts.arr.size(); ++i) {
    if (i) body += ",";
    relql::flat_features(spec, contexts.arr[i], row.data());
    append_float_array(body, row.data(), k);
  }
  body += "]}";
  return body;
}

// ---------------------------------------------------------------------------
// HTTP plumbing (libh2o evloop + worker pool)
// ---------------------------------------------------------------------------

std::string health_json(const Service& svc) {
  std::string resp = "{\"status\":\"ok\",\"device\":\"";
  resp += rt::device_name(svc.device);
  resp += "\",\"forwards\":" + std::to_string(svc.forwards.load());
  resp += ",\"gpu_batches\":" + std::to_string(svc.gpu_batches.load());
  resp += ",\"coalesced_forwards\":" + std::to_string(svc.coalesced.load());
  resp += ",\"texts_embedded\":" + std::to_string(svc.embeds.load()) + "}";
  return resp;
}

struct Job;
// Lives in the request's memory pool; its dispose hook is how the loop
// thread learns the request died while a worker still held the job.
struct ReqLink {
  Job* job;
};

struct Job {
  h2o_multithread_message_t msg{};   // must stay the first member
  h2o_req_t* req = nullptr;          // loop-thread only; null once req died
  ReqLink* link = nullptr;           // loop-thread only
  std::string path, body, response;
  const char* content_type = "application/json";
  int status = 200;
};

Service* g_svc = nullptr;
h2o_multithread_receiver_t g_recv{};

std::mutex g_jobs_mu;
std::condition_variable g_jobs_cv;
std::deque<Job*> g_jobs;

void run_job(Job* job) {
  const auto t0 = std::chrono::steady_clock::now();
  try {
    const relql::JsonValue req = relql::json_parse(job->body);
    const auto t1 = std::chrono::steady_clock::now();
    job->response = job->path == "/v1/forward"
                        ? handle_forward(*g_svc, decode_forward_json(req), false)
                        : handle_flat(req);
    const auto t2 = std::chrono::steady_clock::now();
    auto ms = [](auto a, auto b) {
      return std::chrono::duration<double, std::milli>(b - a).count();
    };
    std::fprintf(stderr, "[serve] %s parse=%.0fms handle=%.0fms in=%zuB out=%zuB\n",
                 job->path.c_str(), ms(t0, t1), ms(t1, t2), job->body.size(),
                 job->response.size());
  } catch (const std::exception& e) {
    job->status = 400;
    job->response = "{\"error\":\"" + json_escape(e.what()) + "\"}";
  }
}

void worker_loop() {
  for (;;) {
    Job* job;
    {
      std::unique_lock<std::mutex> lk(g_jobs_mu);
      g_jobs_cv.wait(lk, [] { return !g_jobs.empty(); });
      job = g_jobs.front();
      g_jobs.pop_front();
    }
    run_job(job);
    h2o_multithread_send_message(&g_recv, &job->msg);
  }
}

void send_body(h2o_req_t* req, int status, const std::string& body,
               const char* content_type) {
  req->res.status = status;
  req->res.reason = status == 200   ? "OK"
                    : status == 404 ? "Not Found"
                                    : "Bad Request";
  h2o_add_header(&req->pool, &req->res.headers, H2O_TOKEN_CONTENT_TYPE,
                 nullptr, content_type, strlen(content_type));
  h2o_send_inline(req, body.data(), body.size());   // copies into req pool
}

void send_json(h2o_req_t* req, int status, const std::string& body) {
  send_body(req, status, body, "application/json");
}

// Runs on the loop thread whenever workers post finished jobs.
void on_job_done(h2o_multithread_receiver_t*, h2o_linklist_t* messages) {
  while (!h2o_linklist_is_empty(messages)) {
    h2o_multithread_message_t* m = H2O_STRUCT_FROM_MEMBER(
        h2o_multithread_message_t, link, messages->next);
    h2o_linklist_unlink(&m->link);
    Job* job = reinterpret_cast<Job*>(m);
    if (job->link) job->link->job = nullptr;   // neutralize the dispose hook
    if (job->req)
      send_body(job->req, job->status, job->response,
                job->status == 200 ? job->content_type : "application/json");
    delete job;
  }
}

void on_req_link_dispose(void* p) {
  ReqLink* link = static_cast<ReqLink*>(p);
  if (link->job) {   // request died first; the job must not touch it
    link->job->req = nullptr;
    link->job->link = nullptr;
  }
}

int on_req(h2o_handler_t*, h2o_req_t* req) {
  const std::string method(req->method.base, req->method.len);
  const std::string path(req->path_normalized.base, req->path_normalized.len);
  if (method == "GET" && path == "/health") {
    send_json(req, 200, health_json(*g_svc));
    return 0;
  }
  if (method == "POST" && (path == "/v1/forward" ||
                           path == "/v1/flat_features")) {
    Job* job = new Job;
    job->req = req;
    job->path = path;
    job->body.assign(req->entity.base, req->entity.len);
    job->link = static_cast<ReqLink*>(h2o_mem_alloc_shared(
        &req->pool, sizeof(ReqLink), on_req_link_dispose));
    job->link->job = job;
    {
      std::lock_guard<std::mutex> lock(g_jobs_mu);
      g_jobs.push_back(job);
    }
    g_jobs_cv.notify_one();
    return 0;   // response arrives via on_job_done
  }
  send_json(req, 404, "{\"error\":\"no such endpoint\"}");
  return 0;
}

h2o_accept_ctx_t g_accept_ctx{};

void on_accept(h2o_socket_t* listener, const char* err) {
  if (err) return;
  h2o_socket_t* sock = h2o_evloop_socket_accept(listener);
  if (sock) h2o_accept(&g_accept_ctx, sock);
}

}  // namespace

int main(int argc, char** argv) {
  int port = 8500;
  std::string device_arg = "auto", minilm_dir, preload;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto val = [&](const char* what) -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "%s needs a value\n", what);
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--port") port = std::atoi(val("--port").c_str());
    else if (a == "--device") device_arg = val("--device");
    else if (a == "--minilm-dir") minilm_dir = val("--minilm-dir");
    else if (a == "--preload") preload = val("--preload");
    else {
      std::fprintf(stderr,
                   "usage: rt_serve [--port N] [--device auto|cpu|mps|cuda] "
                   "[--minilm-dir DIR] [--preload MODEL_URI]\n");
      return a == "--help" || a == "-h" ? 0 : 2;
    }
  }

  Service svc;
  if (device_arg == "auto") {
    svc.device = rt::device_available(rt::Device::MPS)    ? rt::Device::MPS
                 : rt::device_available(rt::Device::CUDA) ? rt::Device::CUDA
                                                          : rt::Device::CPU;
  } else if (device_arg == "cpu") {
    svc.device = rt::Device::CPU;
  } else if (device_arg == "mps") {
    svc.device = rt::Device::MPS;
  } else if (device_arg == "cuda") {
    svc.device = rt::Device::CUDA;
  } else {
    std::fprintf(stderr, "unknown --device %s\n", device_arg.c_str());
    return 2;
  }
  if (svc.device != rt::Device::CPU && !rt::device_available(svc.device)) {
    std::fprintf(stderr, "device %s is not available in this build/machine\n",
                 rt::device_name(svc.device));
    return 2;
  }

  {
    std::string dir = minilm_dir;
    if (dir.empty())
      if (const char* env = std::getenv("RELATIVEDB_MINILM_DIR")) dir = env;
    std::string err;
    svc.encoder = minilm::MiniLM::load(dir, &err);
    if (!svc.encoder) {
      std::fprintf(stderr, "cannot load the MiniLM text encoder: %s\n",
                   err.c_str());
      return 1;
    }
  }
  if (!preload.empty()) svc.model_for(preload);
  g_svc = &svc;
  const int n_prep = env_int("RT_SERVE_PREP_THREADS", 2);
  for (int i = 0; i < n_prep; ++i)
    std::thread(prep_loop, std::ref(svc)).detach();
  // One blocks thread per device lane: on CUDA each thread leases its own
  // stream slot (RT_CUDA_SLOTS), so forwards overlap on the device.
  const int n_blocks = env_int("RT_SERVE_BLOCKS_THREADS", 2);
  for (int i = 0; i < n_blocks; ++i)
    std::thread(blocks_loop, std::ref(svc)).detach();
  const int n_workers = env_int("RT_SERVE_WORKERS", 8);
  for (int i = 0; i < n_workers; ++i) std::thread(worker_loop).detach();

  signal(SIGPIPE, SIG_IGN);   // a client hanging up must not kill the server

  static h2o_globalconf_t config;
  h2o_config_init(&config);
  h2o_hostconf_t* hostconf = h2o_config_register_host(
      &config, h2o_iovec_init(H2O_STRLIT("default")), 65535);
  h2o_pathconf_t* pathconf = h2o_config_register_path(hostconf, "/", 0);
  h2o_handler_t* handler = h2o_create_handler(pathconf, sizeof(*handler));
  handler->on_req = on_req;

  static h2o_context_t ctx;
  h2o_context_init(&ctx, h2o_evloop_create(), &config);
  h2o_multithread_register_receiver(ctx.queue, &g_recv, on_job_done);
  g_accept_ctx.ctx = &ctx;
  g_accept_ctx.hosts = config.hosts;

  int server_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (server_fd < 0) { perror("socket"); return 1; }
  int one = 1;
  setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  addr.sin_port = htons((uint16_t)port);
  if (bind(server_fd, (sockaddr*)&addr, sizeof(addr)) < 0) {
    perror("bind");
    return 1;
  }
  if (listen(server_fd, 128) < 0) { perror("listen"); return 1; }
  h2o_socket_t* listener =
      h2o_evloop_socket_create(ctx.loop, server_fd, H2O_SOCKET_FLAG_DONT_READ);
  h2o_socket_read_start(listener, on_accept);
  std::fprintf(stderr, "[rt_serve] listening on :%d (device %s, h2o, %d workers)\n",
               port, rt::device_name(svc.device), n_workers);

  while (h2o_evloop_run(ctx.loop, INT32_MAX) == 0) {
  }
  return 0;
}
