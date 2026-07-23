#include "rt.hpp"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <thread>

#include "rt_internal.hpp"
#include "rt_math.hpp"

#if defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)
#define RT_SDOT 1
#include <arm_neon.h>
#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif
#endif

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

  // Scale companions produced by rt_quantize ("<name>.q_scale" for Q8,
  // "<name>.q4_scale" for Q4) are attached to their base tensor below, not
  // surfaced as tensors of their own.
  auto is_scale_companion = [&](const std::string& name) {
    for (const char* suf : {".q_scale", ".q4_scale"}) {
      size_t len = std::strlen(suf);
      if (name.size() > len &&
          name.compare(name.size() - len, len, suf) == 0 &&
          entries.count(name.substr(0, name.size() - len)))
        return true;
    }
    return false;
  };
  auto read_raw = [&](const HeaderEntry& e, std::vector<uint8_t>& dst) {
    dst.resize((size_t)(e.end - e.begin));
    f.seekg(data_start + e.begin);
    f.read(reinterpret_cast<char*>(dst.data()), (std::streamsize)dst.size());
  };

  std::unordered_map<std::string, Tensor> out;
  std::vector<uint16_t> tmp;
  for (auto& [name, e] : entries) {
    if (is_scale_companion(name)) continue;
    Tensor t;
    t.shape = e.shape;
    int64_t n = t.numel();
    f.seekg(data_start + e.begin);
    if (e.dtype == "BF16") {
      t.data.resize(n);
      tmp.resize(n);
      f.read(reinterpret_cast<char*>(tmp.data()), n * 2);
      for (int64_t i = 0; i < n; i++) t.data[i] = bf16_to_f32(tmp[i]);
    } else if (e.dtype == "F32") {
      t.data.resize(n);
      f.read(reinterpret_cast<char*>(t.data.data()), n * 4);
    } else if (e.dtype == "F16") {
      // kept quantized-resident; compute paths dequantize in-kernel
      t.qtype = (uint8_t)WType::F16;
      read_raw(e, t.qdata);
    } else if (e.dtype == "I8") {
      auto sit = entries.find(name + ".q_scale");
      if (sit == entries.end() || sit->second.dtype != "F32")
        throw std::runtime_error("I8 tensor " + name + " missing .q_scale");
      t.qtype = (uint8_t)WType::Q8;
      read_raw(e, t.qdata);
      read_raw(sit->second, t.qscale);
    } else if (e.dtype == "U8") {
      // Q4: packed nibbles, stored shape [out, in/2]; logical [out, in]
      auto sit = entries.find(name + ".q4_scale");
      if (sit == entries.end() || sit->second.dtype != "F16")
        throw std::runtime_error("U8 tensor " + name + " missing .q4_scale");
      t.qtype = (uint8_t)WType::Q4;
      if (t.shape.size() != 2)
        throw std::runtime_error("Q4 tensor " + name + " must be 2-D");
      t.shape[1] *= 2;
      read_raw(e, t.qdata);
      read_raw(sit->second, t.qscale);
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

namespace {

// Runtime check for the ARM int8 matrix-multiply extension (SMMLA). The
// SMMLA kernel is compiled with a per-function target attribute, so the
// library still loads on chips without i8mm (e.g. M1) — it just takes the
// SDOT path there. Detected once.
bool cpu_has_i8mm() {
#if defined(RT_SDOT)
  static const bool v = [] {
    if (const char* e = std::getenv("RT_NO_I8MM"))   // force the SDOT path
      if (e[0] == '1') return false;
#if defined(__APPLE__)
    int r = 0;
    size_t s = sizeof r;
    if (sysctlbyname("hw.optional.arm.FEAT_I8MM", &r, &s, nullptr, 0) != 0)
      return false;
    return r != 0;
#elif defined(__ARM_FEATURE_MATMUL_INT8)
    return true;
#else
    return false;
#endif
  }();
  return v;
#else
  return false;
#endif
}

// Repack an int8 [out,in] weight (row-major) into SMMLA panels:
//   packed[(p*(in/8) + kb)*16 + 0..7]  = row(2p)[kb*8 : kb*8+8]
//   packed[(p*(in/8) + kb)*16 + 8..15] = row(2p+1)[kb*8 : kb*8+8]
// so one vld1q_s8 feeds a full 2x8 SMMLA operand. Requires out even, in%8==0.
void pack_q8_smmla(const int8_t* src, int8_t* dst, int out, int in) {
  const int KB = in / 8;
  for (int p = 0; p < out / 2; p++)
    for (int kb = 0; kb < KB; kb++) {
      int8_t* d = dst + ((size_t)p * KB + kb) * 16;
      std::memcpy(d, src + (size_t)(2 * p) * in + kb * 8, 8);
      std::memcpy(d + 8, src + (size_t)(2 * p + 1) * in + kb * 8, 8);
    }
}

}  // namespace

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
    if (w.qtype != (uint8_t)WType::F32)
      throw std::runtime_error(p + ".weight must be fp32 (encoders/decoders "
                               "are never quantized)");
    l.w = w.data.data();
    l.out = static_cast<int>(w.shape[0]);
    l.in = static_cast<int>(w.shape[1]);
    if (bias) l.b = T(p + ".bias").data.data();
    return l;
  };
  auto weight = [&](const std::string& p) {
    Tensor& w = T(p + ".weight");
    Weight ww;
    ww.type = (WType)w.qtype;
    ww.out = static_cast<int>(w.shape[0]);
    ww.in = static_cast<int>(w.shape[1]);
    if (ww.type == WType::F32) {
      ww.f32 = w.data.data();
    } else {
      ww.q = w.qdata.data();
      ww.qs = w.qscale.empty() ? nullptr : w.qscale.data();
    }
    return ww;
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
      // one GEMM; y row layout = [q | k | v | g], row stride 4*d. Rows are
      // independent in every payload format, so stacking is concatenation.
      const char* names[4] = {"wq", "wk", "wv", "wg"};
      Weight parts[4];
      for (int p = 0; p < 4; p++) parts[p] = weight(ap + names[p]);
      const WType wt = parts[0].type;
      for (int p = 1; p < 4; p++)
        if (parts[p].type != wt)
          throw std::runtime_error(ap + ": mixed weight dtypes in qkvg");
      at.wqkvg.type = wt;
      at.wqkvg.out = 4 * kDModel;
      at.wqkvg.in = kDModel;
      if (wt == WType::F32) {
        at.wqkvg_f32.resize((size_t)4 * kDModel * kDModel);
        for (int p = 0; p < 4; p++)
          std::memcpy(&at.wqkvg_f32[(size_t)p * kDModel * kDModel],
                      parts[p].f32, (size_t)kDModel * kDModel * 4);
        at.wqkvg.f32 = at.wqkvg_f32.data();
      } else {
        const size_t rb = row_bytes(wt, kDModel) * kDModel;   // per matrix
        const size_t sb = scale_bytes(wt, kDModel) * kDModel;
        at.wqkvg_q.resize(4 * rb);
        at.wqkvg_s.resize(4 * sb);
        for (int p = 0; p < 4; p++) {
          std::memcpy(&at.wqkvg_q[p * rb], parts[p].q, rb);
          if (sb) std::memcpy(&at.wqkvg_s[p * sb], parts[p].qs, sb);
        }
        at.wqkvg.q = at.wqkvg_q.data();
        at.wqkvg.qs = sb ? at.wqkvg_s.data() : nullptr;
      }
      at.wo = weight(ap + "wo");
      at.q_norm = T(ap + "q_norm.scale").data.data();
      at.k_norm = T(ap + "k_norm.scale").data.data();
      at.head_scale = T(ap + "scale").data.data();  // [1,8,1,1] contiguous
      blk.norm[a] = T(pre + "norms." + kAttnName[a] + ".scale").data.data();
    }
    blk.norm[3] = T(pre + "norms.ffn.scale").data.data();
    blk.w1 = weight(pre + "ffn.w1");
    blk.w2 = weight(pre + "ffn.w2");
    blk.w3 = weight(pre + "ffn.w3");
  }
  m.norm_out = T("norm_out.scale").data.data();
  m.dec_number = lin("dec_dict.number", true);
  m.dec_text = lin("dec_dict.text", true);   // [384,512] weight, [384] bias
  return m;
}

void Model::save(const std::string& path) const {
  // Deterministic key order makes checkpoints reproducible and keeps diffs in
  // metadata tooling stable.  Full-model training deliberately accepts and
  // emits fp32 weights; a quantized inference checkpoint has discarded the
  // precision required for optimization and must not be fine-tuned in place.
  std::vector<std::string> keys;
  keys.reserve(store.size());
  for (const auto& [key, tensor] : store) {
    if (tensor.qtype != (uint8_t)WType::F32)
      throw std::runtime_error(
          "cannot save a trainable checkpoint containing quantized tensor " + key);
    keys.push_back(key);
  }
  std::sort(keys.begin(), keys.end());
  uint64_t offset = 0;
  std::ostringstream header;
  header << '{';
  for (size_t i = 0; i < keys.size(); i++) {
    if (i) header << ',';
    const Tensor& t = store.at(keys[i]);
    const uint64_t bytes = (uint64_t)t.data.size() * sizeof(float);
    header << '"' << keys[i] << "\":{\"dtype\":\"F32\",\"shape\":[";
    for (size_t d = 0; d < t.shape.size(); d++) {
      if (d) header << ',';
      header << t.shape[d];
    }
    header << "],\"data_offsets\":[" << offset << ',' << offset + bytes
           << "]}";
    offset += bytes;
  }
  header << '}';
  std::string json = header.str();
  while (json.size() % 8) json.push_back(' ');
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("cannot create " + path);
  const uint64_t hlen = json.size();
  out.write(reinterpret_cast<const char*>(&hlen), sizeof(hlen));
  out.write(json.data(), (std::streamsize)json.size());
  for (const std::string& key : keys) {
    const Tensor& t = store.at(key);
    out.write(reinterpret_cast<const char*>(t.data.data()),
              (std::streamsize)(t.data.size() * sizeof(float)));
  }
  if (!out) throw std::runtime_error("failed writing " + path);
}

// ---------------------------------------------------------------------------
// primitives
// ---------------------------------------------------------------------------
namespace {

struct Scratch {                 // per-worker attention buffers, reused
  std::vector<float> qb, kb, vb, sc, ob, tmp;
  float denom[detail::kQTile];
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

// y[rows,out] = x[rows,in] @ W^T (+ b). Accelerate sgemm is internally
// threaded; the portable GEMM is parallelized here over row chunks.
void matmul(const float* x, const Linear& l, float* y, int rows) {
#if RT_ACCELERATE
  math::gemm_nt(x, l.w, y, rows, l.out, l.in, l.in, l.in, l.out);
#else
  constexpr int kChunk = 32;
  int chunks = (rows + kChunk - 1) / kChunk;
  Pool::get().run(Pool::get().size(), chunks, [&](int, int c) {
    int r0 = c * kChunk, r1 = std::min(r0 + kChunk, rows);
    math::gemm_nt(x + (size_t)r0 * l.in, l.w, y + (size_t)r0 * l.out, r1 - r0,
                  l.out, l.in, l.in, l.in, l.out);
  });
#endif
  if (l.b) {
    for (int r = 0; r < rows; r++)
      math::vadd(y + (size_t)r * l.out, l.b, y + (size_t)r * l.out, l.out);
  }
}

#ifdef RT_SDOT
inline float q8_absmax(const float* x, int n) {
  float32x4_t mx = vdupq_n_f32(0.f);
  for (int k = 0; k < n; k += 4)
    mx = vmaxnmq_f32(mx, vabsq_f32(vld1q_f32(x + k)));
  return vmaxnmvq_f32(mx);
}

inline int8x8_t quantize_q8x8(const float* x, float inv) {
  const float32x4_t scale = vdupq_n_f32(inv);
  const int32x4_t lo = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x), scale));
  const int32x4_t hi = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x + 4), scale));
  return vqmovn_s16(vcombine_s16(vqmovn_s32(lo), vqmovn_s32(hi)));
}

// True int8 throughput for Q8 weights: activations are quantized per row
// (absmax/127, like llama.cpp's Q8_0 activations), then y = (sx*sw) *
// (xq . wq) with the int8 dot products running on the NEON sdot units
// (vdotq_s32: 16 MACs/instruction). Weight rows stream from DRAM as int8 —
// half the traffic of the fp16 path, a quarter of fp32 — and the fp32
// weights never exist at all. Loop order keeps a 4-row weight strip in L1
// across a 16-row activation block.
struct Q8RowActivations {
  int rows = 0, K = 0;
  std::vector<int8_t> q;
  std::vector<float> scales;
};

Q8RowActivations quantize_q8_rows(const float* x, int rows, int K) {
  Q8RowActivations a;
  a.rows = rows;
  a.K = K;
  a.q.resize((size_t)rows * K);
  a.scales.resize(rows);
  Pool::get().run(Pool::get().size(), rows, [&](int, int r) {
    const float* xr = x + (size_t)r * K;
    const float amax = q8_absmax(xr, K);
    const float s = amax > 0.f ? amax / 127.f : 1.f;
    a.scales[r] = s;
    const float inv = 1.f / s;
    int8_t* q = &a.q[(size_t)r * K];
    for (int k = 0; k < K; k += 8)
      vst1_s8(q + k, quantize_q8x8(xr + k, inv));
  });
  return a;
}

void matmul_q8_sdot_packed(const Q8RowActivations& a, const Weight& w,
                           float* y) {
  assert(a.K == w.in);
  const int rows = a.rows, K = a.K;          // 512/2048: multiples of 16
  const float* sw = reinterpret_cast<const float*>(w.qs);
  const int8_t* wq = reinterpret_cast<const int8_t*>(w.q);
  constexpr int kNTile = 32, kRBlk = 16;
  const int tiles = (w.out + kNTile - 1) / kNTile;
  Pool::get().run(Pool::get().size(), tiles, [&](int, int t) {
    const int n0 = t * kNTile, n1 = std::min(n0 + kNTile, w.out);
    for (int r0 = 0; r0 < rows; r0 += kRBlk) {
      const int r1 = std::min(r0 + kRBlk, rows);
      for (int n = n0; n < n1; n += 4) {
        const int8_t* w0 = wq + (size_t)n * K;
        const int8_t* w1 = w0 + K;
        const int8_t* w2 = w1 + K;
        const int8_t* w3 = w2 + K;
        for (int r = r0; r < r1; r++) {
          const int8_t* xr = &a.q[(size_t)r * K];
          int32x4_t a0 = vdupq_n_s32(0), a1 = a0, a2 = a0, a3 = a0;
          for (int k = 0; k < K; k += 16) {
            int8x16_t xv = vld1q_s8(xr + k);
            a0 = vdotq_s32(a0, xv, vld1q_s8(w0 + k));
            a1 = vdotq_s32(a1, xv, vld1q_s8(w1 + k));
            a2 = vdotq_s32(a2, xv, vld1q_s8(w2 + k));
            a3 = vdotq_s32(a3, xv, vld1q_s8(w3 + k));
          }
          float* yr = y + (size_t)r * w.out + n;
          const float sr = a.scales[r];
          yr[0] = sr * sw[n] * (float)vaddvq_s32(a0);
          yr[1] = sr * sw[n + 1] * (float)vaddvq_s32(a1);
          yr[2] = sr * sw[n + 2] * (float)vaddvq_s32(a2);
          yr[3] = sr * sw[n + 3] * (float)vaddvq_s32(a3);
        }
      }
    }
  });
}

void matmul_q8_sdot(const float* x, const Weight& w, float* y, int rows) {
  auto a = quantize_q8_rows(x, rows, w.in);
  matmul_q8_sdot_packed(a, w, y);
}

// ---- i8mm (SMMLA) path ----------------------------------------------------
// SMMLA does a full 2x8 . 8x2 -> 2x2 int32 matrix-multiply-accumulate per
// instruction (vs SDOT's four 4-lane dots), so one instruction advances a
// 2x2 output block by 8 along K. Both operands are pre-interleaved into 2x8
// panels: weights once on first use (pack_q8_smmla), activations per forward.
// This function carries a per-function i8mm target attribute so the rest of
// the TU stays baseline — it is only ever called when cpu_has_i8mm() packed
// the weights, so chips without i8mm never reach it.
//
// Micro-kernel: 4 activation rows (2 pairs) x 8 output cols (4 pairs) held in
// 8 int32x4 accumulators (16 int32-lane accumulators — enough independent
// chains to saturate the SMMLA units and amortize the panel loads).
__attribute__((target("arch=armv8.6-a+i8mm")))
void smmla_stripe(const int8_t* Apack, const float* sx, const int8_t* Wpack,
                  const float* sw, float* y, int Mp, int M, int N, int K,
                  int n0, int n1) {
  const int KB = K / 8;
  for (int r = 0; r < Mp; r += 4) {
    const int8_t* Ap0 = Apack + (size_t)(r / 2) * KB * 16;       // rows r,   r+1
    const int8_t* Ap1 = Apack + (size_t)(r / 2 + 1) * KB * 16;   // rows r+2, r+3
    for (int n = n0; n < n1; n += 8) {
      const int8_t* Wp0 = Wpack + (size_t)(n / 2) * KB * 16;
      const int8_t* Wp1 = Wp0 + (size_t)KB * 16;
      const int8_t* Wp2 = Wp1 + (size_t)KB * 16;
      const int8_t* Wp3 = Wp2 + (size_t)KB * 16;
      int32x4_t c00 = vdupq_n_s32(0), c01 = c00, c02 = c00, c03 = c00;
      int32x4_t c10 = c00, c11 = c00, c12 = c00, c13 = c00;
      for (int kb = 0; kb < KB; kb++) {
        int8x16_t a0 = vld1q_s8(Ap0 + kb * 16), a1 = vld1q_s8(Ap1 + kb * 16);
        int8x16_t w0 = vld1q_s8(Wp0 + kb * 16), w1 = vld1q_s8(Wp1 + kb * 16),
                  w2 = vld1q_s8(Wp2 + kb * 16), w3 = vld1q_s8(Wp3 + kb * 16);
        c00 = vmmlaq_s32(c00, a0, w0); c01 = vmmlaq_s32(c01, a0, w1);
        c02 = vmmlaq_s32(c02, a0, w2); c03 = vmmlaq_s32(c03, a0, w3);
        c10 = vmmlaq_s32(c10, a1, w0); c11 = vmmlaq_s32(c11, a1, w1);
        c12 = vmmlaq_s32(c12, a1, w2); c13 = vmmlaq_s32(c13, a1, w3);
      }
      // lanes of each 2x2 block: [ (row_even . col_even), (row_even . col_odd),
      //                            (row_odd  . col_even), (row_odd  . col_odd) ]
      const int32x4_t C[2][4] = {{c00, c01, c02, c03}, {c10, c11, c12, c13}};
      for (int i = 0; i < 2; i++) {
        const int re = r + 2 * i, ro = re + 1;
        if (re >= M) break;                     // padded tile tail
        for (int j = 0; j < 4; j++) {
          const int ce = n + 2 * j, co = ce + 1;
          const int32x4_t v = C[i][j];
          y[(size_t)re * N + ce] = sx[re] * sw[ce] * (float)vgetq_lane_s32(v, 0);
          y[(size_t)re * N + co] = sx[re] * sw[co] * (float)vgetq_lane_s32(v, 1);
          if (ro < M) {
            y[(size_t)ro * N + ce] = sx[ro] * sw[ce] * (float)vgetq_lane_s32(v, 2);
            y[(size_t)ro * N + co] = sx[ro] * sw[co] * (float)vgetq_lane_s32(v, 3);
          }
        }
      }
    }
  }
}

// Quantize directly into pair-packed activation panels, then run smmla_stripe
// over N stripes. Writing the final layout here avoids materializing a
// row-major int8 panel and reading it back in a separate packing pass.
// The weight repacking is built once and cached (w.q8_smmla); w.q itself is
// left row-major so the GPU backend and SDOT path keep their canonical view.
struct Q8SmmlaActivations {
  int rows = 0, K = 0, Mp = 0;
  std::unique_ptr<int8_t[]> q;
  std::vector<float> scales;
};

Q8SmmlaActivations quantize_q8_smmla(const float* x, int rows, int K) {
  Q8SmmlaActivations a;
  a.rows = rows;
  a.K = K;
  a.Mp = (rows + 3) & ~3;                   // pad rows to a multiple of 4
  a.scales.resize(a.Mp, 0.f);
  const int KB = K / 8;
  // Every panel byte is assigned below, including at most three padded rows,
  // so avoid zero-filling the full buffer before overwriting its real rows.
  a.q.reset(new int8_t[(size_t)(a.Mp / 2) * KB * 16]);
  Pool::get().run(Pool::get().size(), a.Mp / 2, [&](int, int p) {
    int8_t* d = a.q.get() + (size_t)p * KB * 16;
    for (int rp = 0; rp < 2; rp++) {
      const int r = 2 * p + rp;
      if (r >= rows) {
        for (int kb = 0; kb < KB; kb++)
          vst1_s8(d + kb * 16 + rp * 8, vdup_n_s8(0));
        continue;
      }
      const float* xr = x + (size_t)r * K;
      const float amax = q8_absmax(xr, K);
      const float s = amax > 0.f ? amax / 127.f : 1.f;
      a.scales[r] = s;
      const float inv = 1.f / s;
      for (int kb = 0; kb < KB; kb++)
        vst1_s8(d + kb * 16 + rp * 8, quantize_q8x8(xr + kb * 8, inv));
    }
  });
  return a;
}

void matmul_q8_smmla_packed(const Q8SmmlaActivations& a, const Weight& w,
                            float* y) {
  assert(a.K == w.in);
  const int K = w.in, N = w.out;
  if (!w.q8_smmla) {
    static std::mutex mu;
    std::lock_guard<std::mutex> lk(mu);
    if (!w.q8_smmla) {                       // double-checked under the lock
      auto packed = std::make_shared<std::vector<int8_t>>((size_t)N * K);
      pack_q8_smmla(reinterpret_cast<const int8_t*>(w.q), packed->data(), N, K);
      w.q8_smmla = packed;
    }
  }
  const int8_t* Wpack = w.q8_smmla->data();
  const float* sw = reinterpret_cast<const float*>(w.qs);
  constexpr int kStripe = 64;               // multiple of 8
  const int nstripes = (N + kStripe - 1) / kStripe;
  Pool::get().run(Pool::get().size(), nstripes, [&](int, int st) {
    const int n0 = st * kStripe, n1 = std::min(n0 + kStripe, N);
    smmla_stripe(a.q.get(), a.scales.data(), Wpack, sw, y, a.Mp, a.rows, N, K,
                 n0, n1);
  });
}

void matmul_q8_smmla(const float* x, const Weight& w, float* y, int rows) {
  auto a = quantize_q8_smmla(x, rows, w.in);
  matmul_q8_smmla_packed(a, w, y);
}

// w1 and w3 consume the same normalized FFN input. Quantize/pack that panel
// once and feed both projections instead of repeating the activation pass.
bool matmul_q4_pair(const float* x, const Weight& w0, float* y0,
                    const Weight& w1, float* y1, int rows);

bool matmul_q8_pair(const float* x, const Weight& w0, float* y0,
                    const Weight& w1, float* y1, int rows) {
  if (w0.type == WType::Q4 || w1.type == WType::Q4)
    return matmul_q4_pair(x, w0, y0, w1, y1, rows);
  if (w0.type != WType::Q8 || w1.type != WType::Q8 || w0.in != w1.in)
    return false;
  if (cpu_has_i8mm() && w0.out % 8 == 0 && w1.out % 8 == 0 &&
      w0.in % 8 == 0) {
    auto a = quantize_q8_smmla(x, rows, w0.in);
    matmul_q8_smmla_packed(a, w0, y0);
    matmul_q8_smmla_packed(a, w1, y1);
    return true;
  }
  if (w0.out % 4 == 0 && w1.out % 4 == 0 && w0.in % 16 == 0) {
    auto a = quantize_q8_rows(x, rows, w0.in);
    matmul_q8_sdot_packed(a, w0, y0);
    matmul_q8_sdot_packed(a, w1, y1);
    return true;
  }
  return false;
}

// ---- Q4 SDOT path ---------------------------------------------------------
// Q4 rows are asymmetric per 32-group: w = s_g * nib + m_g. With per-row
// int8 activations (xq, scale sx) and exact fp32 per-group activation sums
// S_g precomputed at quantization time,
//   y = sum_g [ sx * s_g * (xq_g . nib_g) + m_g * S_g ]
// so the bulk multiply runs int8 x int8 on the sdot units and the nibble
// payload streams straight from DRAM (an eighth of fp32 traffic) with no
// dequant pass. Activations are stored deinterleaved per 32-group (16
// even-index bytes then 16 odd) so the two nibble planes of one 16-byte
// weight load dot against contiguous vectors. Activation quantization noise
// matches the Q8 path (absmax/127); the m_g * S_g term stays exact.
// Escape hatch env RT_NO_Q4_SDOT forces the tile-dequant path.
bool use_q4_sdot() {
  static const bool v = [] {
    const char* e = std::getenv("RT_NO_Q4_SDOT");
    return !(e && e[0] == '1');
  }();
  return v;
}

struct Q4Activations {
  int rows = 0, K = 0;
  std::vector<int8_t> q;               // [rows, K] deinterleaved per 32-group
  std::vector<float> scales;           // per row
  std::vector<float> gsums;            // [rows, K/32] fp32 group sums of x
};

Q4Activations quantize_q4_rows(const float* x, int rows, int K) {
  Q4Activations a;
  a.rows = rows;
  a.K = K;
  a.q.resize((size_t)rows * K);
  a.scales.resize(rows);
  a.gsums.resize((size_t)rows * (K / 32));
  Pool::get().run(Pool::get().size(), rows, [&](int, int r) {
    const float* xr = x + (size_t)r * K;
    const float amax = q8_absmax(xr, K);
    const float s = amax > 0.f ? amax / 127.f : 1.f;
    a.scales[r] = s;
    const float inv = 1.f / s;
    int8_t* q = &a.q[(size_t)r * K];
    float* gs = &a.gsums[(size_t)r * (K / 32)];
    for (int g = 0; g < K / 32; g++) {
      const float* xg = xr + g * 32;
      int8_t tmp[32];
      for (int k = 0; k < 32; k += 8)
        vst1_s8(tmp + k, quantize_q8x8(xg + k, inv));
      const int8x16x2_t de = vld2q_s8(tmp);       // split even/odd indices
      vst1q_s8(q + g * 32, de.val[0]);
      vst1q_s8(q + g * 32 + 16, de.val[1]);
      float32x4_t sum = vld1q_f32(xg);
      for (int k = 4; k < 32; k += 4) sum = vaddq_f32(sum, vld1q_f32(xg + k));
      gs[g] = vaddvq_f32(sum);
    }
  });
  return a;
}

void matmul_q4_sdot_packed(const Q4Activations& a, const Weight& w, float* y) {
  assert(a.K == w.in);
  const int rows = a.rows, K = a.K, G = K / 32;   // K: multiple of 32
  const size_t rb = (size_t)K / 2, sb = (size_t)G * 4;
  constexpr int kNTile = 32, kRBlk = 16, kGMax = 64;  // K <= 2048
  const int tiles = (w.out + kNTile - 1) / kNTile;
  Pool::get().run(Pool::get().size(), tiles, [&](int, int t) {
    const int n0 = t * kNTile, n1 = std::min(n0 + kNTile, w.out);
    const int8x16_t mask = vdupq_n_s8(0x0f);
    // hoist fp16 scale conversion once per tile (32 rows x G groups)
    float sg[kNTile][kGMax], mg[kNTile][kGMax];
    for (int n = n0; n < n1; n++) {
      const uint16_t* sh =
          reinterpret_cast<const uint16_t*>(w.qs + (size_t)n * sb);
      for (int g = 0; g < G; g++) {
        sg[n - n0][g] = half_to_float(sh[2 * g]);
        mg[n - n0][g] = half_to_float(sh[2 * g + 1]);
      }
    }
    // The asymmetric min term sum_g m_g * S_g is a dense [rows, G] @ [N, G]^T
    // product: seed this y tile with it through the fp32 GEMM (AMX on Apple)
    // so the int8 loop below only adds the sdot part.
    math::gemm_nt(a.gsums.data(), &mg[0][0], y + n0, rows, n1 - n0, G, G,
                  kGMax, w.out);
    // same blocking as the q8 sdot kernel: a 4-row weight strip stays hot in
    // L1 across a 16-row activation block, activation loads shared by the
    // strip. Per-group dots stay in int32 lanes and are folded with one
    // vectorized fma per (row, group); the only horizontal reduce is one
    // vaddvq per output element.
    for (int r0 = 0; r0 < rows; r0 += kRBlk) {
      const int r1 = std::min(r0 + kRBlk, rows);
      for (int n = n0; n < n1; n += 4) {
        const uint8_t* w0 = w.q + (size_t)n * rb;
        const uint8_t* w1 = w0 + rb;
        const uint8_t* w2 = w1 + rb;
        const uint8_t* w3 = w2 + rb;
        const float *s0 = sg[n - n0], *s1 = sg[n - n0 + 1];
        const float *s2 = sg[n - n0 + 2], *s3 = sg[n - n0 + 3];
        for (int r = r0; r < r1; r++) {
          const int8_t* xr = &a.q[(size_t)r * K];
          float32x4_t f0 = vdupq_n_f32(0.f), f1 = f0, f2 = f0, f3 = f0;
          for (int g = 0; g < G; g++) {
            const int8x16_t xe = vld1q_s8(xr + g * 32);
            const int8x16_t xo = vld1q_s8(xr + g * 32 + 16);
            const uint8x16_t b0 = vld1q_u8(w0 + g * 16);
            const uint8x16_t b1 = vld1q_u8(w1 + g * 16);
            const uint8x16_t b2 = vld1q_u8(w2 + g * 16);
            const uint8x16_t b3 = vld1q_u8(w3 + g * 16);
            int32x4_t d0 = vdotq_s32(
                vdotq_s32(vdupq_n_s32(0), xe,
                          vandq_s8(vreinterpretq_s8_u8(b0), mask)),
                xo, vreinterpretq_s8_u8(vshrq_n_u8(b0, 4)));
            int32x4_t d1 = vdotq_s32(
                vdotq_s32(vdupq_n_s32(0), xe,
                          vandq_s8(vreinterpretq_s8_u8(b1), mask)),
                xo, vreinterpretq_s8_u8(vshrq_n_u8(b1, 4)));
            int32x4_t d2 = vdotq_s32(
                vdotq_s32(vdupq_n_s32(0), xe,
                          vandq_s8(vreinterpretq_s8_u8(b2), mask)),
                xo, vreinterpretq_s8_u8(vshrq_n_u8(b2, 4)));
            int32x4_t d3 = vdotq_s32(
                vdotq_s32(vdupq_n_s32(0), xe,
                          vandq_s8(vreinterpretq_s8_u8(b3), mask)),
                xo, vreinterpretq_s8_u8(vshrq_n_u8(b3, 4)));
            f0 = vfmaq_n_f32(f0, vcvtq_f32_s32(d0), s0[g]);
            f1 = vfmaq_n_f32(f1, vcvtq_f32_s32(d1), s1[g]);
            f2 = vfmaq_n_f32(f2, vcvtq_f32_s32(d2), s2[g]);
            f3 = vfmaq_n_f32(f3, vcvtq_f32_s32(d3), s3[g]);
          }
          const float sx = a.scales[r];
          float* yr = y + (size_t)r * w.out + n;
          yr[0] += sx * vaddvq_f32(f0);
          yr[1] += sx * vaddvq_f32(f1);
          yr[2] += sx * vaddvq_f32(f2);
          yr[3] += sx * vaddvq_f32(f3);
        }
      }
    }
  });
}

void matmul_q4_sdot(const float* x, const Weight& w, float* y, int rows) {
  auto a = quantize_q4_rows(x, rows, w.in);
  matmul_q4_sdot_packed(a, w, y);
}

// w1/w3 share the same normalized FFN input: quantize the activation panel
// once. Declared above matmul_q8_pair, which routes Q4 pairs here.
bool matmul_q4_pair(const float* x, const Weight& w0, float* y0,
                    const Weight& w1, float* y1, int rows) {
  if (w0.type != WType::Q4 || w1.type != WType::Q4 || w0.in != w1.in ||
      !use_q4_sdot() || w0.out % 4 != 0 || w1.out % 4 != 0 ||
      w0.in % 32 != 0 || w0.in / 32 > 64)
    return false;
  auto a = quantize_q4_rows(x, rows, w0.in);
  matmul_q4_sdot_packed(a, w0, y0);
  matmul_q4_sdot_packed(a, w1, y1);
  return true;
}

// F16 weights stream from DRAM as IEEE half — half the fp32 traffic — and
// widen in-register (vcvt) right next to the fp32 FMA, so neither a scratch
// dequant pass nor the fp32 weight row ever exists in memory. Activations
// stay fp32, so numerics match the dequantize-then-GEMM reference up to
// summation order. Same loop order as the SDOT kernel: a 4-row weight strip
// stays hot in L1 across the activation block.
//
// Only used for micro-batches (rows <= kF16NeonMaxRows): measured on M4,
// streaming NEON wins 2.7x at M=1 and 1.2x at M=4, but the tile-dequant +
// Accelerate path wins 3x by M=16 and 12x by M=128 — AMX outruns the NEON
// FMA units as soon as the dequant cost is amortized over enough rows.
// Escape hatch env RT_NO_F16_NEON forces tile-dequant (mirrors RT_NO_I8MM).
constexpr int kF16NeonMaxRows = 4;
bool use_f16_neon() {
  static const bool v = [] {
    const char* e = std::getenv("RT_NO_F16_NEON");
    return !(e && e[0] == '1');
  }();
  return v;
}

void matmul_f16_neon(const float* x, const Weight& w, float* y, int rows) {
  const float16_t* wq = reinterpret_cast<const float16_t*>(w.q);
  const int K = w.in;                        // 512/2048: multiples of 8
  constexpr int kNTile = 32, kRBlk = 16;
  const int tiles = (w.out + kNTile - 1) / kNTile;
  Pool::get().run(Pool::get().size(), tiles, [&](int, int t) {
    const int n0 = t * kNTile, n1 = std::min(n0 + kNTile, w.out);
    for (int r0 = 0; r0 < rows; r0 += kRBlk) {
      const int r1 = std::min(r0 + kRBlk, rows);
      for (int n = n0; n < n1; n += 4) {
        const float16_t* w0 = wq + (size_t)n * K;
        const float16_t* w1 = w0 + K;
        const float16_t* w2 = w1 + K;
        const float16_t* w3 = w2 + K;
        for (int r = r0; r < r1; r++) {
          const float* xr = x + (size_t)r * K;
          // two accumulators per output row: independent FMA chains for the
          // low/high half of each 8-wide weight load
          float32x4_t a0 = vdupq_n_f32(0.f), b0 = a0, a1 = a0, b1 = a0;
          float32x4_t a2 = a0, b2 = a0, a3 = a0, b3 = a0;
          for (int k = 0; k < K; k += 8) {
            const float32x4_t xlo = vld1q_f32(xr + k);
            const float32x4_t xhi = vld1q_f32(xr + k + 4);
            const float16x8_t v0 = vld1q_f16(w0 + k);
            const float16x8_t v1 = vld1q_f16(w1 + k);
            const float16x8_t v2 = vld1q_f16(w2 + k);
            const float16x8_t v3 = vld1q_f16(w3 + k);
            a0 = vfmaq_f32(a0, xlo, vcvt_f32_f16(vget_low_f16(v0)));
            b0 = vfmaq_f32(b0, xhi, vcvt_high_f32_f16(v0));
            a1 = vfmaq_f32(a1, xlo, vcvt_f32_f16(vget_low_f16(v1)));
            b1 = vfmaq_f32(b1, xhi, vcvt_high_f32_f16(v1));
            a2 = vfmaq_f32(a2, xlo, vcvt_f32_f16(vget_low_f16(v2)));
            b2 = vfmaq_f32(b2, xhi, vcvt_high_f32_f16(v2));
            a3 = vfmaq_f32(a3, xlo, vcvt_f32_f16(vget_low_f16(v3)));
            b3 = vfmaq_f32(b3, xhi, vcvt_high_f32_f16(v3));
          }
          float* yr = y + (size_t)r * w.out + n;
          yr[0] = vaddvq_f32(vaddq_f32(a0, b0));
          yr[1] = vaddvq_f32(vaddq_f32(a1, b1));
          yr[2] = vaddvq_f32(vaddq_f32(a2, b2));
          yr[3] = vaddvq_f32(vaddq_f32(a3, b3));
        }
      }
    }
  });
}
#endif

// y[rows,out] = x[rows,in] @ W^T for a quantization-aware Weight.
// F32 delegates to the plain GEMM. Q8 on ARM runs the int8 sdot kernel
// (int8 activations x int8 weights). Other quantized types stay resident:
// each worker dequantizes a 64-output-row tile of W into a per-thread fp32
// scratch (hot in L1/L2) and runs the GEMM for that column stripe of y —
// DRAM weight traffic is the quantized payload, the fp32 tile never leaves
// cache. Tiles are independent, so they parallelize on the pool.
void matmul_w(const float* x, const Weight& w, float* y, int rows,
              float beta = 0.f) {
  if (w.type == WType::F32) {
    math::gemm_nt(x, w.f32, y, rows, w.out, w.in, w.in, w.in, w.out, beta);
    return;
  }
  assert(beta == 0.f);
#ifdef RT_SDOT
  if (w.type == WType::Q8 && cpu_has_i8mm() && w.out % 8 == 0 && w.in % 8 == 0) {
    matmul_q8_smmla(x, w, y, rows);         // i8mm (weights packed lazily)
    return;
  }
  if (w.type == WType::Q8 && w.out % 4 == 0 && w.in % 16 == 0) {
    matmul_q8_sdot(x, w, y, rows);
    return;
  }
  if (w.type == WType::Q4 && use_q4_sdot() && w.out % 4 == 0 &&
      w.in % 32 == 0 && w.in / 32 <= 64) {
    matmul_q4_sdot(x, w, y, rows);
    return;
  }
  if (w.type == WType::F16 && rows <= kF16NeonMaxRows && use_f16_neon() &&
      w.out % 4 == 0 && w.in % 8 == 0) {
    matmul_f16_neon(x, w, y, rows);
    return;
  }
#endif
  // Wider output tiles amortize dequantization and BLAS submission once the
  // activation panel is large; keep more independent stripes for latency
  // shapes where a wide GEMM cannot make use of them.
  const int tile = rows >= 256 ? 128 : 64;
  const int tiles = (w.out + tile - 1) / tile;
  const size_t rb = row_bytes(w.type, w.in);
  const size_t sb = scale_bytes(w.type, w.in);
  Pool::get().run(Pool::get().size(), tiles, [&](int, int t) {
    thread_local std::vector<float> scratch;
    if (scratch.size() < (size_t)tile * w.in)
      scratch.resize((size_t)tile * w.in);
    const int n0 = t * tile, n1 = std::min(n0 + tile, w.out);
    for (int n = n0; n < n1; n++) {
      float* dst = &scratch[(size_t)(n - n0) * w.in];
      const uint8_t* q = w.q + (size_t)n * rb;
      switch (w.type) {
        case WType::F16: dequant_row_f16(q, dst, w.in); break;
        case WType::Q8: dequant_row_q8(q, w.qs + (size_t)n * sb, dst, w.in); break;
        default: dequant_row_q4(q, w.qs + (size_t)n * sb, dst, w.in); break;
      }
    }
    math::gemm_nt(x, scratch.data(), y + n0, rows, n1 - n0, w.in, w.in, w.in,
                  w.out);
  });
}

// out = rmsnorm(x) * scale, fp32, row of length n.
inline void rmsnorm(const float* x, const float* scale, float* out, int n) {
  float ss = 0.f;
  for (int i = 0; i < n; i++) ss += x[i] * x[i];
  float inv = 1.0f / std::sqrt(ss / n + kEps);
  for (int i = 0; i < n; i++) out[i] = x[i] * inv * scale[i];
}

}  // namespace

namespace detail {

float bf16_round(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  uint32_t lsb = (u >> 16) & 1;
  u += 0x7fffu + lsb;
  u &= 0xffff0000u;
  std::memcpy(&f, &u, 4);
  return f;
}

// ---------------------------------------------------------------------------
// batch preparation (device-independent, always on CPU)
// ---------------------------------------------------------------------------
Prepared prepare(const Model& m, const Batch& batch, Output& out,
                 bool debug_taps) {
  const int B = batch.B, S = batch.S, D = kDModel;
  Prepared prep;
  prep.B = B; prep.S = S;
  out.B = B; out.S = S;
  out.sort_idxs.resize((size_t)B * S);
  out.sorted_is_target.resize((size_t)B * S);
  out.yhat_number.resize((size_t)B * S);

  // ---- stable sort by column id (padding last), per batch row -------------
  std::vector<int64_t> node((size_t)B * S), colid((size_t)B * S), tabid((size_t)B * S),
      sem((size_t)B * S);
  std::vector<int64_t> f2p((size_t)B * S * kMaxF2p);
  std::vector<uint8_t> tgt((size_t)B * S);
  std::vector<uint8_t>& pad = prep.pad;
  pad.resize((size_t)B * S);
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
  prep.g_col.resize(B); prep.g_feat.resize(B); prep.g_nbr.resize(B);
  std::vector<Groups>& g_col = prep.g_col;
  std::vector<Groups>& g_feat = prep.g_feat;
  std::vector<Groups>& g_nbr = prep.g_nbr;
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
  auto tiles = [&](const std::vector<Groups>& gs, std::vector<Work>& w) {
    for (int b = 0; b < B; b++)
      for (int gi = 0; gi < gs[b].n(); gi++) {
        int nq = gs[b].qoff[gi + 1] - gs[b].qoff[gi];
        int nk = gs[b].koff[gi + 1] - gs[b].koff[gi];
        // log(clamp_min(bf16(count),1)) — mirrors kv_sizes.bfloat16() upstream
        float logkv = std::log(std::max(bf16_round((float)nk), 1.0f));
        for (int q0 = 0; q0 < nq; q0 += kQTile)
          w.push_back({b, gi, q0, std::min(q0 + kQTile, nq), logkv});
      }
  };
  tiles(g_col, prep.work[0]);
  tiles(g_feat, prep.work[1]);
  tiles(g_nbr, prep.work[2]);

  // ---- embeddings ---------------------------------------------------------
  const size_t BS = (size_t)B * S;
  prep.x.assign(BS * D, 0.f);
  std::vector<float>& x = prep.x;
  std::vector<float> tmp(BS * D);
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
        math::vadd(&x[i * D], row, &x[i * D], D);
      } else {
        math::vadd(&x[i * D], m.mask_emb[t], &x[i * D], D);
      }
    }
  }
  if (debug_taps) out.x_embed = x;
  return prep;
}

// ---------------------------------------------------------------------------
// CPU backend: transformer blocks + output head
// ---------------------------------------------------------------------------
void run_blocks_cpu(const Model& m, Prepared& prep, Output& out, int n_threads,
                    bool debug_taps, bool want_text_head,
                    bool want_target_features) {
  const int B = prep.B, S = prep.S, D = kDModel;
  if (n_threads <= 0)
    n_threads = std::max(1u, std::thread::hardware_concurrency());
  const size_t BS = (size_t)B * S;
  std::vector<float>& x = prep.x;

  constexpr int C4 = 4 * kDModel;      // qkvg row stride; q|k|v|g at 0,D,2D,3D
  std::vector<float> xn(BS * D), qkvg(BS * (size_t)C4), att(BS * D), proj(BS * D);
  std::vector<float> ffa(BS * kDFF), ffb(BS * kDFF);

  auto parallel_for = [&](int total, auto&& fn) {   // fn(worker, item)
    std::function<void(int, int)> f = fn;
    Pool::get().run(n_threads, total, f);
  };
  auto norm_rows = [&](const float* in, const float* scale, float* dst,
                       size_t rows) {
    constexpr int kChunk = 16;
    if (rows < 256) {
      for (size_t i = 0; i < rows; i++)
        rmsnorm(in + i * D, scale, dst + i * D, D);
      return;
    }
    const int chunks = ((int)rows + kChunk - 1) / kChunk;
    parallel_for(chunks, [&](int, int c) {
      const size_t r0 = (size_t)c * kChunk;
      const size_t r1 = std::min(r0 + kChunk, rows);
      for (size_t i = r0; i < r1; i++)
        rmsnorm(in + i * D, scale, dst + i * D, D);
    });
  };
  std::vector<Scratch> scratch(std::min(n_threads, Pool::get().size()));

  for (int blk_i = 0; blk_i < kBlocks; blk_i++) {
    const Block& blk = m.blocks[blk_i];
    for (int a = 0; a < 3; a++) {
      const Attn& at = blk.attn[a];
      const auto& gs = a == 0 ? prep.g_col : a == 1 ? prep.g_feat : prep.g_nbr;
      const auto& wl = prep.work[a];
      // pre-norm
      norm_rows(x.data(), blk.norm[a], xn.data(), BS);
      matmul_w(xn.data(), at.wqkvg, qkvg.data(), (int)BS);
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
        const float logkv = W.logkv;
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
          math::gemm_nt(sp.qb.data() + h * kHeadDim, sp.kb.data() + h * kHeadDim,
                        sp.sc.data(), tq, nk, kHeadDim, D, D, nk);
          for (int r = 0; r < tq; r++) {      // stable two-pass softmax rows
            float* srow = &sp.sc[(size_t)r * nk];
            float mx = math::maxv(srow, nk);
            math::vsadd(srow, -mx, srow, nk);
            math::vexp_inplace(srow, nk);
            sp.denom[r] = math::sum(srow, nk);
          }
          math::gemm_nn(sp.sc.data(), sp.vb.data() + h * kHeadDim,
                        sp.ob.data() + h * kHeadDim, tq, kHeadDim, nk, nk, D, D);
          for (int r = 0; r < tq; r++) {      // normalize after the PV GEMM
            float inv = 1.0f / sp.denom[r];
            math::vsmul(&sp.ob[(size_t)r * D + h * kHeadDim], inv,
                        &sp.ob[(size_t)r * D + h * kHeadDim], kHeadDim);
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
          math::vneg(grow, sp.tmp.data(), D);
          math::vexp_inplace(sp.tmp.data(), D);
          math::vsadd(sp.tmp.data(), 1.0f, sp.tmp.data(), D);
          math::vdiv(src, sp.tmp.data(), dst, D);
          math::vsmul(dst, 2.0f, dst, D);
        }
      });
      if (at.wo.type == WType::F32) {
        matmul_w(att.data(), at.wo, x.data(), (int)BS, 1.f);
      } else {
        matmul_w(att.data(), at.wo, proj.data(), (int)BS);
        math::vadd(x.data(), proj.data(), x.data(), BS * D);
      }
    }
    // FFN: x += w2( silu(w1 xn) * w3 xn )
    norm_rows(x.data(), blk.norm[3], xn.data(), BS);
#ifdef RT_SDOT
    if (!matmul_q8_pair(xn.data(), blk.w1, ffa.data(), blk.w3, ffb.data(),
                        (int)BS))
#endif
    {
      matmul_w(xn.data(), blk.w1, ffa.data(), (int)BS);
      matmul_w(xn.data(), blk.w3, ffb.data(), (int)BS);
    }
    // silu(ffa) * ffb per row: silu(x) = x/(1+exp(-x))
    parallel_for((int)BS, [&](int w, int i) {
      Scratch& sp = scratch[w];
      if (sp.tmp.size() < (size_t)kDFF) sp.tmp.resize(kDFF);
      float* fa = &ffa[(size_t)i * kDFF];
      math::vneg(fa, sp.tmp.data(), kDFF);
      math::vexp_inplace(sp.tmp.data(), kDFF);
      math::vsadd(sp.tmp.data(), 1.0f, sp.tmp.data(), kDFF);
      math::vdiv(fa, sp.tmp.data(), fa, kDFF);
      math::vmul(fa, &ffb[(size_t)i * kDFF], fa, kDFF);
    });
    if (blk.w2.type == WType::F32) {
      matmul_w(ffa.data(), blk.w2, x.data(), (int)BS, 1.f);
    } else {
      matmul_w(ffa.data(), blk.w2, proj.data(), (int)BS);
      math::vadd(x.data(), proj.data(), x.data(), BS * D);
    }
    if (blk_i == 0 && debug_taps) out.x_block0 = x;
  }

  // ---- output norm + number head -----------------------------------------
  norm_rows(x.data(), m.norm_out, xn.data(), BS);
  for (size_t i = 0; i < BS; i++) {
    out.yhat_number[i] =
        m.dec_number.b[0] + math::dot(&xn[i * D], m.dec_number.w, D);
  }

  // ---- text head (additive; mirrors number head: norm_out then Linear) ----
  // Same norm_out(x) input the number head consumes; only the target cells are
  // populated (all the ABI reads), the rest stay 0. dec_text.w is [384,512],
  // row d begins at d*D.
  if (want_text_head) {
    out.yhat_text.assign(BS * (size_t)kDText, 0.f);
    for (size_t i = 0; i < BS; i++) {
      if (!out.sorted_is_target[i]) continue;
      float* dst = &out.yhat_text[i * kDText];
      for (int d = 0; d < kDText; d++)
        dst[d] = m.dec_text.b[d] + math::dot(&xn[i * D], &m.dec_text.w[(size_t)d * D], D);
    }
  }
  if (want_target_features) {
    out.target_features.assign((size_t)B * D, 0.f);
    for (int b = 0; b < B; b++)
      for (int s = 0; s < S; s++) {
        const size_t i = (size_t)b * S + s;
        if (!out.sorted_is_target[i]) continue;
        math::vadd(&out.target_features[(size_t)b * D], &xn[i * D],
                   &out.target_features[(size_t)b * D], D);
      }
  }
}

}  // namespace detail

// ---------------------------------------------------------------------------
// public entry points / device dispatch
// ---------------------------------------------------------------------------
const char* device_name(Device d) {
  switch (d) {
    case Device::CPU: return "cpu";
    case Device::MPS: return "mps";
    case Device::CUDA: return "cuda";
  }
  return "?";
}

bool device_available(Device d) {
  switch (d) {
    case Device::CPU:
      return true;
    case Device::MPS:
#ifdef RT_METAL
      return detail::metal_available();
#else
      return false;
#endif
    case Device::CUDA:
#ifdef RT_CUDA
      return detail::cuda_available();
#else
      return false;
#endif
  }
  return false;
}

Output forward(const Model& m, const Batch& batch, const ForwardOpts& opts) {
  Output out;
  // RT_FORWARD_PROFILE=1: coarse prepare-vs-blocks wall split to stderr.
  const bool fprof = std::getenv("RT_FORWARD_PROFILE") != nullptr;
  auto t0 = std::chrono::steady_clock::now();
  detail::Prepared prep = detail::prepare(m, batch, out, opts.debug_taps);
  if (fprof) {
    fprintf(stderr, "[rt-forward] prepare %.1fms\n",
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count());
    t0 = std::chrono::steady_clock::now();
  }
  struct BlocksTimer {
    bool on; std::chrono::steady_clock::time_point t;
    ~BlocksTimer() {
      if (on)
        fprintf(stderr, "[rt-forward] blocks  %.1fms\n",
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t).count());
    }
  } bt{fprof, t0};
  switch (opts.device) {
    case Device::CPU:
      detail::run_blocks_cpu(m, prep, out, opts.n_threads, opts.debug_taps,
                             opts.want_text_head, opts.want_target_features);
      return out;
    case Device::MPS:
#ifdef RT_METAL
      if (opts.want_text_head)
        throw std::runtime_error("rt: text decoder output is CPU-only; use "
                                 "target features for Metal fine-tuning");
      detail::run_blocks_metal(m, prep, out, opts.debug_taps,
                               opts.want_target_features);
      return out;
#else
      break;
#endif
    case Device::CUDA:
#ifdef RT_CUDA
      detail::run_blocks_cuda(m, prep, out, opts.debug_taps);
      return out;
#else
      break;
#endif
  }
  throw std::runtime_error(std::string("rt: backend not compiled in: ") +
                           device_name(opts.device));
}

Output forward(const Model& m, const Batch& batch, int n_threads,
               bool debug_taps) {
  ForwardOpts o;
  o.n_threads = n_threads;
  o.debug_taps = debug_taps;
  return forward(m, batch, o);
}

}  // namespace rt
