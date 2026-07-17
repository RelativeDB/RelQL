#include "rt.hpp"

#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <fstream>
#include <functional>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <thread>

namespace rt {

// ---------------------------------------------------------------------------
// safetensors loader: 8-byte LE header length, JSON header, raw data.
// The header is flat {"name": {"dtype":"BF16","shape":[..],"data_offsets":[a,b]}}
// so a tiny purpose-built scanner is enough — no JSON dependency.
// ---------------------------------------------------------------------------
namespace {

inline float bf16_to_f32(uint16_t v) {
  uint32_t u = static_cast<uint32_t>(v) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// Round fp32 to bf16 (round-to-nearest-even) and back — used to mirror the
// Python side's `.bfloat16()` cast of kv counts.
inline float bf16_round(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  uint32_t lsb = (u >> 16) & 1;
  u += 0x7fffu + lsb;
  u &= 0xffff0000u;
  std::memcpy(&f, &u, 4);
  return f;
}

struct HeaderEntry {
  std::string dtype;
  std::vector<int64_t> shape;
  int64_t begin = 0, end = 0;
};

// Minimal scanner for the safetensors flat header.
std::unordered_map<std::string, HeaderEntry> parse_header(const std::string& h) {
  std::unordered_map<std::string, HeaderEntry> out;
  size_t i = 0;
  auto skip_ws = [&] { while (i < h.size() && isspace((unsigned char)h[i])) i++; };
  auto parse_str = [&]() -> std::string {
    assert(h[i] == '"');
    size_t j = h.find('"', ++i);
    std::string s = h.substr(i, j - i);
    i = j + 1;
    return s;
  };
  skip_ws();
  assert(h[i] == '{');
  i++;
  while (true) {
    skip_ws();
    if (h[i] == '}') break;
    if (h[i] == ',') { i++; continue; }
    std::string name = parse_str();
    skip_ws(); assert(h[i] == ':'); i++; skip_ws();
    // object
    assert(h[i] == '{'); i++;
    HeaderEntry e;
    while (true) {
      skip_ws();
      if (h[i] == '}') { i++; break; }
      if (h[i] == ',') { i++; continue; }
      std::string key = parse_str();
      skip_ws(); assert(h[i] == ':'); i++; skip_ws();
      if (key == "dtype") {
        e.dtype = parse_str();
      } else if (key == "shape" || key == "data_offsets") {
        assert(h[i] == '['); i++;
        std::vector<int64_t> vals;
        while (h[i] != ']') {
          if (h[i] == ',' || isspace((unsigned char)h[i])) { i++; continue; }
          vals.push_back(strtoll(&h[i], nullptr, 10));
          while (h[i] != ',' && h[i] != ']') i++;
        }
        i++;
        if (key == "shape") e.shape = vals;
        else { e.begin = vals[0]; e.end = vals[1]; }
      } else {  // __metadata__ values etc. — skip a string
        if (h[i] == '"') parse_str();
        else while (h[i] != ',' && h[i] != '}') i++;
      }
    }
    if (name != "__metadata__") out[name] = e;
  }
  return out;
}

}  // namespace

std::unordered_map<std::string, Tensor> load_safetensors(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  uint64_t hlen = 0;
  f.read(reinterpret_cast<char*>(&hlen), 8);
  std::string header(hlen, '\0');
  f.read(header.data(), hlen);
  auto entries = parse_header(header);
  int64_t data_start = 8 + static_cast<int64_t>(hlen);

  std::unordered_map<std::string, Tensor> out;
  std::vector<uint16_t> tmp;
  for (auto& [name, e] : entries) {
    Tensor t;
    t.shape = e.shape;
    int64_t n = t.numel();
    t.data.resize(n);
    f.seekg(data_start + e.begin);
    if (e.dtype == "BF16") {
      tmp.resize(n);
      f.read(reinterpret_cast<char*>(tmp.data()), n * 2);
      for (int64_t i = 0; i < n; i++) t.data[i] = bf16_to_f32(tmp[i]);
    } else if (e.dtype == "F32") {
      f.read(reinterpret_cast<char*>(t.data.data()), n * 4);
    } else {
      throw std::runtime_error("unsupported dtype " + e.dtype + " for " + name);
    }
    out[name] = std::move(t);
  }
  return out;
}

// ---------------------------------------------------------------------------
// model wiring
// ---------------------------------------------------------------------------
static const char* kSem[4] = {"number", "text", "datetime", "boolean"};
static const char* kAttnName[3] = {"col", "feat", "nbr"};

Model Model::load(const std::string& path) {
  Model m;
  m.store = load_safetensors(path);
  auto T = [&](const std::string& k) -> Tensor& {
    auto it = m.store.find(k);
    if (it == m.store.end()) throw std::runtime_error("missing tensor " + k);
    return it->second;
  };
  auto lin = [&](const std::string& p, bool bias) {
    Linear l;
    Tensor& w = T(p + ".weight");
    l.w = w.data.data();
    l.out = static_cast<int>(w.shape[0]);
    l.in = static_cast<int>(w.shape[1]);
    if (bias) l.b = T(p + ".bias").data.data();
    return l;
  };
  for (int t = 0; t < 4; t++) {
    m.enc[t] = lin(std::string("enc_dict.") + kSem[t], true);
    m.norm_enc[t] = T(std::string("norm_dict.") + kSem[t] + ".scale").data.data();
    m.mask_emb[t] = T(std::string("mask_embs.") + kSem[t]).data.data();
  }
  m.enc_col_name = lin("enc_dict.col_name", true);
  m.norm_col_name = T("norm_dict.col_name.scale").data.data();
  for (int b = 0; b < kBlocks; b++) {
    std::string pre = "blocks." + std::to_string(b) + ".";
    Block& blk = m.blocks[b];
    for (int a = 0; a < 3; a++) {
      std::string ap = pre + "attns." + kAttnName[a] + ".";
      Attn& at = blk.attn[a];
      // stack wq/wk/wv/wg along the output dim so the four projections run as
      // one GEMM; y row layout = [q | k | v | g], row stride 4*d
      at.wqkvg.resize((size_t)4 * kDModel * kDModel);
      const char* names[4] = {"wq", "wk", "wv", "wg"};
      for (int p = 0; p < 4; p++) {
        Linear l = lin(ap + names[p], false);
        std::memcpy(&at.wqkvg[(size_t)p * kDModel * kDModel], l.w,
                    (size_t)kDModel * kDModel * 4);
      }
      at.wo = lin(ap + "wo", false);
      at.q_norm = T(ap + "q_norm.scale").data.data();
      at.k_norm = T(ap + "k_norm.scale").data.data();
      at.head_scale = T(ap + "scale").data.data();  // [1,8,1,1] contiguous
      blk.norm[a] = T(pre + "norms." + kAttnName[a] + ".scale").data.data();
    }
    blk.norm[3] = T(pre + "norms.ffn.scale").data.data();
    blk.w1 = lin(pre + "ffn.w1", false);
    blk.w2 = lin(pre + "ffn.w2", false);
    blk.w3 = lin(pre + "ffn.w3", false);
  }
  m.norm_out = T("norm_out.scale").data.data();
  m.dec_number = lin("dec_dict.number", true);
  return m;
}

// ---------------------------------------------------------------------------
// primitives
// ---------------------------------------------------------------------------
namespace {

// y[rows,out] = x[rows,in] @ W^T (+ b). Row-major, Accelerate sgemm.
void matmul(const float* x, const Linear& l, float* y, int rows) {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, rows, l.out, l.in, 1.0f,
              x, l.in, l.w, l.in, 0.0f, y, l.out);
  if (l.b) {
    for (int r = 0; r < rows; r++)
      vDSP_vadd(y + (size_t)r * l.out, 1, l.b, 1, y + (size_t)r * l.out, 1, l.out);
  }
}

// out = rmsnorm(x) * scale, fp32, row of length n.
inline void rmsnorm(const float* x, const float* scale, float* out, int n) {
  float ss = 0.f;
  for (int i = 0; i < n; i++) ss += x[i] * x[i];
  float inv = 1.0f / std::sqrt(ss / n + kEps);
  for (int i = 0; i < n; i++) out[i] = x[i] * inv * scale[i];
}

inline float sigmoidf(float v) { return 1.0f / (1.0f + std::exp(-v)); }

// Query-groups for masked attention: every query in a group attends to the
// same key list, so scores/output are computed as small per-head GEMMs over
// query tiles instead of streaming key-by-key per query.
struct Groups {
  std::vector<int> q;            // flattened query idxs (sorted positions)
  std::vector<int> qoff{0};      // per group
  std::vector<int> k;            // flattened key idxs (shared by the group)
  std::vector<int> koff{0};
  int n() const { return (int)qoff.size() - 1; }
  void add(const std::vector<int>& qs, const std::vector<int>& ks) {
    q.insert(q.end(), qs.begin(), qs.end());
    k.insert(k.end(), ks.begin(), ks.end());
    qoff.push_back((int)q.size());
    koff.push_back((int)k.size());
  }
};

constexpr int kQTile = 64;       // queries per attention work item

struct Scratch {                 // per-worker attention buffers, reused
  std::vector<float> qb, kb, vb, sc, ob, tmp;
  float denom[kQTile];
};

// Persistent thread pool: workers park on a condition variable between jobs
// instead of being spawned per parallel_for (72 spawns/forward otherwise).
// The caller participates as worker 0; items are claimed via atomic counter.
class Pool {
 public:
  static Pool& get() {
    static Pool p;
    return p;
  }
  int size() const { return (int)workers_.size() + 1; }
  void run(int max_workers, int total, const std::function<void(int, int)>& fn) {
    int participants = std::min({max_workers, total, size()});
    if (participants <= 1) {
      for (int i = 0; i < total; i++) fn(0, i);
      return;
    }
    {
      std::lock_guard<std::mutex> lk(mu_);
      job_ = &fn;
      total_ = total;
      next_.store(0, std::memory_order_relaxed);
      active_ = participants - 1;
      done_ = 0;
      gen_++;
    }
    cv_.notify_all();
    for (int i; (i = next_.fetch_add(1, std::memory_order_relaxed)) < total;)
      fn(0, i);
    std::unique_lock<std::mutex> lk(mu_);
    cv_done_.wait(lk, [&] { return done_ == active_; });
    job_ = nullptr;
  }

 private:
  Pool() {
    int n = (int)std::max(1u, std::thread::hardware_concurrency()) - 1;
    for (int w = 0; w < n; w++) workers_.emplace_back([this, w] { worker(w + 1); });
  }
  ~Pool() {
    {
      std::lock_guard<std::mutex> lk(mu_);
      stop_ = true;
      gen_++;
    }
    cv_.notify_all();
    for (auto& t : workers_) t.join();
  }
  void worker(int id) {
    uint64_t seen = 0;
    while (true) {
      const std::function<void(int, int)>* job;
      int total;
      {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return stop_ || gen_ != seen; });
        seen = gen_;
        if (stop_) return;
        if (id > active_) continue;    // not participating this round
        job = job_;
        total = total_;
      }
      for (int i; (i = next_.fetch_add(1, std::memory_order_relaxed)) < total;)
        (*job)(id, i);
      {
        std::lock_guard<std::mutex> lk(mu_);
        done_++;
      }
      cv_done_.notify_one();
    }
  }
  std::vector<std::thread> workers_;
  std::mutex mu_;
  std::condition_variable cv_, cv_done_;
  const std::function<void(int, int)>* job_ = nullptr;
  std::atomic<int> next_{0};
  int total_ = 0, active_ = 0, done_ = 0;
  uint64_t gen_ = 0;
  bool stop_ = false;
};

}  // namespace

// ---------------------------------------------------------------------------
// forward
// ---------------------------------------------------------------------------
Output forward(const Model& m, const Batch& batch, int n_threads, bool debug_taps) {
  const int B = batch.B, S = batch.S, D = kDModel;
  if (n_threads <= 0)
    n_threads = std::max(1u, std::thread::hardware_concurrency());
  Output out;
  out.B = B; out.S = S;
  out.sort_idxs.resize((size_t)B * S);
  out.sorted_is_target.resize((size_t)B * S);
  out.yhat_number.resize((size_t)B * S);

  // ---- stable sort by column id (padding last), per batch row -------------
  std::vector<int64_t> node((size_t)B * S), colid((size_t)B * S), tabid((size_t)B * S),
      sem((size_t)B * S);
  std::vector<int64_t> f2p((size_t)B * S * kMaxF2p);
  std::vector<uint8_t> pad((size_t)B * S), tgt((size_t)B * S);
  std::vector<float> numv((size_t)B * S), datv((size_t)B * S), boolv((size_t)B * S);
  std::vector<float> textv((size_t)B * S * kDText), colv((size_t)B * S * kDText);
  for (int b = 0; b < B; b++) {
    std::vector<int> order(S);
    std::iota(order.begin(), order.end(), 0);
    const int64_t* ci = &batch.col_idxs[(size_t)b * S];
    const uint8_t* pi = &batch.is_padding[(size_t)b * S];
    std::stable_sort(order.begin(), order.end(), [&](int a, int c) {
      int64_t ka = pi[a] ? INT64_MAX : ci[a];
      int64_t kc = pi[c] ? INT64_MAX : ci[c];
      return ka < kc;
    });
    for (int s = 0; s < S; s++) {
      int src = order[s];
      size_t di = (size_t)b * S + s, si = (size_t)b * S + src;
      out.sort_idxs[di] = src;
      node[di] = batch.node_idxs[si];
      colid[di] = batch.col_idxs[si];
      tabid[di] = batch.table_idxs[si];
      sem[di] = batch.sem_types[si];
      pad[di] = batch.is_padding[si];
      tgt[di] = batch.is_target[si];
      numv[di] = batch.number_v[si];
      datv[di] = batch.datetime_v[si];
      boolv[di] = batch.boolean_v[si];
      std::memcpy(&f2p[di * kMaxF2p], &batch.f2p[si * kMaxF2p], kMaxF2p * 8);
      std::memcpy(&textv[di * kDText], &batch.text_v[si * kDText], kDText * 4);
      std::memcpy(&colv[di * kDText], &batch.col_name_v[si * kDText], kDText * 4);
      out.sorted_is_target[di] = tgt[di];
    }
  }

  // ---- build query-groups for the three attention types -------------------
  // (queries sharing a key list are grouped so attention runs as GEMMs;
  //  every non-pad token lands in exactly one group per type)
  std::vector<Groups> g_col(B), g_feat(B), g_nbr(B);
  for (int b = 0; b < B; b++) {
    const size_t base = (size_t)b * S;
    std::unordered_map<int64_t, std::vector<int>> by_coltab, by_node, nbr_of;
    for (int s = 0; s < S; s++) {
      if (pad[base + s]) continue;
      by_coltab[(colid[base + s] << 32) ^ tabid[base + s]].push_back(s);
      by_node[node[base + s]].push_back(s);
      for (int j = 0; j < kMaxF2p; j++) {
        int64_t p = f2p[(base + s) * kMaxF2p + j];
        if (p >= 0) nbr_of[p].push_back(s);   // key s is a child of node p
      }
    }
    // col: group members attend to each other (queries == keys)
    for (auto& [key, mem] : by_coltab) g_col[b].add(mem, mem);
    // feat: tokens sharing (node, f2p list) share the key list — own row plus
    // rows referenced by the foreign keys (dedup: a parent id equal to own
    // node or a repeated parent must not double-count)
    std::unordered_map<std::string, int> fid;
    std::vector<std::vector<int>> fmem;
    for (int s = 0; s < S; s++) {
      if (pad[base + s]) continue;
      char kb[8 * (1 + kMaxF2p)];
      std::memcpy(kb, &node[base + s], 8);
      std::memcpy(kb + 8, &f2p[(base + s) * kMaxF2p], kMaxF2p * 8);
      auto [it, fresh] = fid.try_emplace(std::string(kb, sizeof kb), (int)fmem.size());
      if (fresh) fmem.emplace_back();
      fmem[it->second].push_back(s);
    }
    std::vector<int> keys;
    for (auto& mem : fmem) {
      keys.clear();
      int s0 = mem[0];
      int64_t own = node[base + s0];
      auto& grp = by_node[own];
      keys.insert(keys.end(), grp.begin(), grp.end());
      int64_t seen[kMaxF2p + 1];
      int nseen = 0;
      seen[nseen++] = own;
      for (int j = 0; j < kMaxF2p; j++) {
        int64_t p = f2p[(size_t)(base + s0) * kMaxF2p + j];
        if (p < 0) continue;
        bool dup = false;
        for (int t = 0; t < nseen; t++) dup |= (seen[t] == p);
        if (dup) continue;
        seen[nseen++] = p;
        auto it = by_node.find(p);
        if (it != by_node.end()) keys.insert(keys.end(), it->second.begin(), it->second.end());
      }
      g_feat[b].add(mem, keys);
    }
    // nbr: tokens of a node attend to its reverse-FK children; nodes without
    // children are skipped (fully-masked queries output zero)
    for (auto& [nid, mem] : by_node) {
      auto it = nbr_of.find(nid);
      if (it != nbr_of.end()) g_nbr[b].add(mem, it->second);
    }
  }

  // ---- attention work items: (batch row, group, query tile) ---------------
  struct Work { int b, g, q0, q1; };
  auto tiles = [&](const std::vector<Groups>& gs) {
    std::vector<Work> w;
    for (int b = 0; b < B; b++)
      for (int gi = 0; gi < gs[b].n(); gi++) {
        int nq = gs[b].qoff[gi + 1] - gs[b].qoff[gi];
        for (int q0 = 0; q0 < nq; q0 += kQTile)
          w.push_back({b, gi, q0, std::min(q0 + kQTile, nq)});
      }
    return w;
  };
  const std::vector<Work> work[3] = {tiles(g_col), tiles(g_feat), tiles(g_nbr)};

  // ---- embeddings ---------------------------------------------------------
  const size_t BS = (size_t)B * S;
  std::vector<float> x(BS * D, 0.f), tmp(BS * D);
  // col-name embedding for every non-pad token
  matmul(colv.data(), m.enc_col_name, tmp.data(), (int)BS);
  for (size_t i = 0; i < BS; i++) {
    if (pad[i]) continue;
    rmsnorm(&tmp[i * D], m.norm_col_name, &x[i * D], D);
  }
  // per-sem-type value encodings / mask embeddings
  std::vector<float> scalar_in(BS);
  for (int t = 0; t < 4; t++) {
    const float* src1 = t == kNumber ? numv.data()
                       : t == kDatetime ? datv.data()
                       : t == kBoolean ? boolv.data() : nullptr;
    if (t == kText) {
      matmul(textv.data(), m.enc[t], tmp.data(), (int)BS);
    } else {
      for (size_t i = 0; i < BS; i++) {
        float v = src1[i];
        scalar_in[i] = std::isnan(v) ? 0.f : v;   // NaN -> 0 like the FIXME path
      }
      matmul(scalar_in.data(), m.enc[t], tmp.data(), (int)BS);
    }
    float row[kDModel];
    for (size_t i = 0; i < BS; i++) {
      if (pad[i] || sem[i] != t) continue;
      if (!tgt[i]) {
        rmsnorm(&tmp[i * D], m.norm_enc[t], row, D);
        vDSP_vadd(&x[i * D], 1, row, 1, &x[i * D], 1, D);
      } else {
        vDSP_vadd(&x[i * D], 1, m.mask_emb[t], 1, &x[i * D], 1, D);
      }
    }
  }
  if (debug_taps) out.x_embed = x;

  // ---- transformer blocks -------------------------------------------------
  constexpr int C4 = 4 * kDModel;      // qkvg row stride; q|k|v|g at 0,D,2D,3D
  std::vector<float> xn(BS * D), qkvg(BS * (size_t)C4), att(BS * D), proj(BS * D);
  std::vector<float> ffa(BS * kDFF), ffb(BS * kDFF);

  auto parallel_for = [&](int total, auto&& fn) {   // fn(worker, item)
    std::function<void(int, int)> f = fn;
    Pool::get().run(n_threads, total, f);
  };
  std::vector<Scratch> scratch(std::min(n_threads, Pool::get().size()));

  for (int blk_i = 0; blk_i < kBlocks; blk_i++) {
    const Block& blk = m.blocks[blk_i];
    for (int a = 0; a < 3; a++) {
      const Attn& at = blk.attn[a];
      const auto& gs = a == 0 ? g_col : a == 1 ? g_feat : g_nbr;
      const auto& wl = work[a];
      // pre-norm
      for (size_t i = 0; i < BS; i++) rmsnorm(&x[i * D], blk.norm[a], &xn[i * D], D);
      Linear fused{at.wqkvg.data(), nullptr, C4, D};
      matmul(xn.data(), fused, qkvg.data(), (int)BS);
      // K-norm depends only on the key token — apply once per (token, head)
      // instead of per (query, key) pair inside the attention loop.
      parallel_for((int)BS, [&](int, int i) {
        float row[kHeadDim];
        for (int h = 0; h < kHeads; h++) {
          float* kr = &qkvg[(size_t)i * C4 + D + h * kHeadDim];
          rmsnorm(kr, at.k_norm, row, kHeadDim);
          std::memcpy(kr, row, sizeof(row));
        }
      });
      // Masked attention per (group, query tile): scores and output are
      // per-head GEMMs over the tile; tokens absent from every group
      // (padding, fully-masked nbr queries) stay zero from the fill below.
      std::fill(att.begin(), att.end(), 0.f);
      parallel_for((int)wl.size(), [&](int w, int wi) {
        const Work& W = wl[wi];
        const Groups& G = gs[W.b];
        const int* qs = &G.q[G.qoff[W.g] + W.q0];
        const int tq = W.q1 - W.q0;
        const int* ks = &G.k[G.koff[W.g]];
        const int nk = G.koff[W.g + 1] - G.koff[W.g];
        const size_t rb = (size_t)W.b * S;
        Scratch& sp = scratch[w];
        if ((int64_t)sp.kb.size() < (int64_t)nk * D) {
          sp.kb.resize((size_t)nk * D);
          sp.vb.resize((size_t)nk * D);
        }
        if ((int64_t)sp.sc.size() < (int64_t)tq * nk) sp.sc.resize((size_t)tq * nk);
        if (sp.qb.empty()) {
          sp.qb.resize((size_t)kQTile * D);
          sp.ob.resize((size_t)kQTile * D);
        }
        // log(clamp_min(bf16(count),1)) — mirrors kv_sizes.bfloat16() upstream
        const float logkv = std::log(std::max(bf16_round((float)nk), 1.0f));
        for (int j = 0; j < nk; j++) {        // gather K/V rows (K pre-normed)
          const float* src = &qkvg[(rb + ks[j]) * C4];
          std::memcpy(&sp.kb[(size_t)j * D], src + D, D * 4);
          std::memcpy(&sp.vb[(size_t)j * D], src + 2 * D, D * 4);
        }
        // gather Q rows: QK-norm + head scale x logkv, 1/head_dim folded in
        for (int r = 0; r < tq; r++) {
          const float* qrow = &qkvg[(rb + qs[r]) * C4];
          for (int h = 0; h < kHeads; h++) {
            float* dst = &sp.qb[(size_t)r * D + h * kHeadDim];
            rmsnorm(qrow + h * kHeadDim, at.q_norm, dst, kHeadDim);
            float qscale = at.head_scale[h] * logkv / kHeadDim;
            for (int d = 0; d < kHeadDim; d++) dst[d] *= qscale;
          }
        }
        for (int h = 0; h < kHeads; h++) {
          cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, tq, nk, kHeadDim,
                      1.0f, sp.qb.data() + h * kHeadDim, D,
                      sp.kb.data() + h * kHeadDim, D, 0.0f, sp.sc.data(), nk);
          for (int r = 0; r < tq; r++) {      // stable two-pass softmax rows
            float* srow = &sp.sc[(size_t)r * nk];
            float mx;
            vDSP_maxv(srow, 1, &mx, nk);
            mx = -mx;
            vDSP_vsadd(srow, 1, &mx, srow, 1, nk);
            int cnt = nk;
            vvexpf(srow, srow, &cnt);
            vDSP_sve(srow, 1, &sp.denom[r], nk);
          }
          cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, tq, kHeadDim,
                      nk, 1.0f, sp.sc.data(), nk, sp.vb.data() + h * kHeadDim,
                      D, 0.0f, sp.ob.data() + h * kHeadDim, D);
          for (int r = 0; r < tq; r++) {      // normalize after the PV GEMM
            float inv = 1.0f / sp.denom[r];
            vDSP_vsmul(&sp.ob[(size_t)r * D + h * kHeadDim], 1, &inv,
                       &sp.ob[(size_t)r * D + h * kHeadDim], 1, kHeadDim);
          }
        }
        // gate = 2*sigmoid(wg(xn)) elementwise, applied on scatter:
        // dst = src * 2/(1+exp(-g))
        if (sp.tmp.size() < (size_t)D) sp.tmp.resize(D);
        for (int r = 0; r < tq; r++) {
          size_t i = rb + qs[r];
          const float* grow = &qkvg[i * C4 + 3 * D];
          const float* src = &sp.ob[(size_t)r * D];
          float* dst = &att[i * D];
          int cnt = D;
          float one = 1.0f, two = 2.0f;
          vDSP_vneg(grow, 1, sp.tmp.data(), 1, D);
          vvexpf(sp.tmp.data(), sp.tmp.data(), &cnt);
          vDSP_vsadd(sp.tmp.data(), 1, &one, sp.tmp.data(), 1, D);
          vDSP_vdiv(sp.tmp.data(), 1, src, 1, dst, 1, D);
          vDSP_vsmul(dst, 1, &two, dst, 1, D);
        }
      });
      matmul(att.data(), at.wo, proj.data(), (int)BS);
      vDSP_vadd(x.data(), 1, proj.data(), 1, x.data(), 1, BS * D);
    }
    // FFN: x += w2( silu(w1 xn) * w3 xn )
    for (size_t i = 0; i < BS; i++) rmsnorm(&x[i * D], blk.norm[3], &xn[i * D], D);
    matmul(xn.data(), blk.w1, ffa.data(), (int)BS);
    matmul(xn.data(), blk.w3, ffb.data(), (int)BS);
    // silu(ffa) * ffb per row: silu(x) = x/(1+exp(-x))
    parallel_for((int)BS, [&](int w, int i) {
      Scratch& sp = scratch[w];
      if (sp.tmp.size() < (size_t)kDFF) sp.tmp.resize(kDFF);
      float* fa = &ffa[(size_t)i * kDFF];
      int cnt = kDFF;
      float one = 1.0f;
      vDSP_vneg(fa, 1, sp.tmp.data(), 1, kDFF);
      vvexpf(sp.tmp.data(), sp.tmp.data(), &cnt);
      vDSP_vsadd(sp.tmp.data(), 1, &one, sp.tmp.data(), 1, kDFF);
      vDSP_vdiv(sp.tmp.data(), 1, fa, 1, fa, 1, kDFF);
      vDSP_vmul(fa, 1, &ffb[(size_t)i * kDFF], 1, fa, 1, kDFF);
    });
    matmul(ffa.data(), blk.w2, proj.data(), (int)BS);
    vDSP_vadd(x.data(), 1, proj.data(), 1, x.data(), 1, BS * D);
    if (blk_i == 0 && debug_taps) out.x_block0 = x;
  }

  // ---- output norm + number head -----------------------------------------
  for (size_t i = 0; i < BS; i++) rmsnorm(&x[i * D], m.norm_out, &xn[i * D], D);
  for (size_t i = 0; i < BS; i++) {
    float acc = m.dec_number.b[0];
    const float* w = m.dec_number.w;
    float dot = 0.f;
    vDSP_dotpr(&xn[i * D], 1, w, 1, &dot, D);
    out.yhat_number[i] = acc + dot;
  }
  return out;
}

}  // namespace rt
