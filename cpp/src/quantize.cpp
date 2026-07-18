// quantize.cpp — RT-J checkpoint quantizer.
//
//   ./rt_quantize <in.safetensors> <out.safetensors> [--type q8|q4|f16]
//
// Every transformer-block projection (qkvg/wo/ffn across the 12 blocks —
// ~99% of parameters) is converted to the requested format; the value/
// col-name encoders, decoder head, norms, biases and mask embeddings stay
// fp32 (input-side error would propagate through all 12 blocks for a
// negligible size win). Formats (see rt_quant.hpp for layouts):
//
//   q8   int8, per-output-row symmetric fp32 scale ("<name>.q_scale")
//   q4   int4, groups of 32 along the input dim, fp16 scale per group
//        (payload: U8 [out, in/2], scales: "<name>.q4_scale" F16)
//   f16  IEEE half payload, no scales
//
// The rt loader keeps these payloads resident and the CPU/MPS compute paths
// dequantize inside the GEMM kernels — quantization here reduces both the
// artifact size and steady-state memory/bandwidth.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "rt.hpp"
#include "rt_quant.hpp"

namespace {

struct OutTensor {
  std::string dtype;                  // "F32", "F16", "I8", "U8"
  std::vector<int64_t> shape;
  std::vector<uint8_t> bytes;
};

// Quantize only the transformer-block projections; see file comment.
bool should_quantize(const std::string& name, const rt::Tensor& t) {
  return t.shape.size() == 2 && t.shape[1] >= 64 &&
         t.shape[1] % rt::kQ4Group == 0 &&
         name.rfind("blocks.", 0) == 0 &&
         name.size() > 7 && name.rfind(".weight") == name.size() - 7;
}

std::string json_escape(const std::string& s) {
  std::string o;
  for (char c : s) {
    if (c == '"' || c == '\\') o += '\\';
    o += c;
  }
  return o;
}

// error accumulators: mean |w_hat - w| / mean |w| per row, worst over rows
struct ErrStat {
  double worst = 0;
  void add(double err, double ref) {
    if (ref > 0) worst = std::max(worst, err / ref);
  }
};

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s <in.safetensors> <out.safetensors> "
                    "[--type q8|q4|f16]\n", argv[0]);
    return 2;
  }
  rt::WType wt = rt::WType::Q8;
  for (int i = 3; i + 1 < argc; i++) {
    if (std::string(argv[i]) == "--type") {
      std::string t = argv[i + 1];
      if (t == "q8") wt = rt::WType::Q8;
      else if (t == "q4") wt = rt::WType::Q4;
      else if (t == "f16") wt = rt::WType::F16;
      else { fprintf(stderr, "unknown --type %s\n", t.c_str()); return 2; }
    }
  }
  auto tensors = rt::load_safetensors(argv[1]);

  // q4 keeps the residual-writing projections (wo, ffn.w2) at q8 — same
  // recipe as llama.cpp's Q4_K_M keeping attn.output/ffn.down higher-bit:
  // their error lands directly on the residual stream every block, and
  // they are only ~1/3 of the quantized bytes.
  auto row_type = [&](const std::string& name) {
    if (wt != rt::WType::Q4) return wt;
    if (name.find(".wo.") != std::string::npos ||
        name.find("ffn.w2.") != std::string::npos)
      return rt::WType::Q8;
    return rt::WType::Q4;
  };

  // std::map: deterministic output order
  std::map<std::string, OutTensor> out;
  int n_quant = 0;
  ErrStat stat;
  for (auto& [name, t] : tensors) {
    if (t.qtype != 0) {
      fprintf(stderr, "input tensor %s is already quantized — quantize from "
                      "the fp32/bf16 checkpoint\n", name.c_str());
      return 1;
    }
    if (!should_quantize(name, t)) {
      OutTensor f{"F32", t.shape, {}};
      f.bytes.resize(t.data.size() * 4);
      std::memcpy(f.bytes.data(), t.data.data(), f.bytes.size());
      out[name] = std::move(f);
      continue;
    }
    const int64_t rows = t.shape[0], cols = t.shape[1];
    n_quant++;
    const rt::WType tw = row_type(name);
    if (tw == rt::WType::F16) {
      OutTensor q{"F16", t.shape, {}};
      q.bytes.resize((size_t)rows * cols * 2);
      auto* h = reinterpret_cast<uint16_t*>(q.bytes.data());
      for (int64_t i = 0; i < rows * cols; i++) {
        h[i] = rt::float_to_half(t.data[i]);
      }
      // f16 round-trip error is ~1e-3 relative; skip per-row stats
      out[name] = std::move(q);
    } else if (tw == rt::WType::Q8) {
      OutTensor q{"I8", t.shape, {}};
      q.bytes.resize((size_t)rows * cols);
      OutTensor sc{"F32", {rows}, {}};
      sc.bytes.resize((size_t)rows * 4);
      auto* scales = reinterpret_cast<float*>(sc.bytes.data());
      for (int64_t r = 0; r < rows; r++) {
        const float* w = &t.data[(size_t)r * cols];
        float amax = 0.f;
        for (int64_t c = 0; c < cols; c++) amax = std::max(amax, std::fabs(w[c]));
        float scale = amax > 0.f ? amax / 127.f : 1.f;
        scales[r] = scale;
        auto* qr = reinterpret_cast<int8_t*>(&q.bytes[(size_t)r * cols]);
        double err = 0, ref = 0;
        for (int64_t c = 0; c < cols; c++) {
          int v = (int)std::lround(w[c] / scale);
          v = std::clamp(v, -127, 127);
          qr[c] = (int8_t)v;
          err += std::fabs(v * scale - w[c]);
          ref += std::fabs(w[c]);
        }
        stat.add(err, ref);
      }
      out[name] = std::move(q);
      out[name + ".q_scale"] = std::move(sc);
    } else {  // Q4_1-style: groups of 32, fp16 (scale, min) per group
      OutTensor q{"U8", {rows, cols / 2}, {}};
      q.bytes.resize((size_t)rows * cols / 2);
      const int64_t groups = cols / rt::kQ4Group;
      OutTensor sc{"F16", {rows, groups * 2}, {}};
      sc.bytes.resize((size_t)rows * groups * 4);
      auto* sh = reinterpret_cast<uint16_t*>(sc.bytes.data());
      for (int64_t r = 0; r < rows; r++) {
        const float* w = &t.data[(size_t)r * cols];
        uint8_t* qr = &q.bytes[(size_t)r * cols / 2];
        double err = 0, ref = 0;
        for (int64_t g = 0; g < groups; g++) {
          const float* wg = w + g * rt::kQ4Group;
          float mn0 = wg[0], mx0 = wg[0];
          for (int i = 1; i < rt::kQ4Group; i++) {
            mn0 = std::min(mn0, wg[i]);
            mx0 = std::max(mx0, wg[i]);
          }
          // clip search: shrinking the range clips outliers but shrinks the
          // step for everything else — pick the min-MSE candidate.
          float mn = mn0, mx = mx0, best = -1.f;
          for (float shrink : {1.f, 0.95f, 0.9f, 0.85f, 0.8f}) {
            const float mid = 0.5f * (mn0 + mx0);
            const float half = 0.5f * (mx0 - mn0) * shrink;
            const float cmn = mid - half;
            const float cs = half > 0.f ? 2.f * half / 15.f : 1.f;
            float mse = 0.f;
            for (int i = 0; i < rt::kQ4Group; i++) {
              int v = std::clamp((int)std::lround((wg[i] - cmn) / cs), 0, 15);
              float d = v * cs + cmn - wg[i];
              mse += d * d;
            }
            if (best < 0.f || mse < best) {
              best = mse;
              mn = cmn;
              mx = mid + half;
            }
          }
          float scale = mx > mn ? (mx - mn) / 15.f : 1.f;
          // store scale/min as the f16 the loader will read back
          uint16_t hs = rt::float_to_half(scale);
          uint16_t hm = rt::float_to_half(mn);
          sh[(r * groups + g) * 2] = hs;
          sh[(r * groups + g) * 2 + 1] = hm;
          scale = rt::half_to_float(hs);
          mn = rt::half_to_float(hm);
          uint8_t* qg = qr + g * rt::kQ4Group / 2;
          for (int i = 0; i < rt::kQ4Group / 2; i++) {
            int v0 = std::clamp((int)std::lround((wg[2 * i] - mn) / scale), 0, 15);
            int v1 = std::clamp((int)std::lround((wg[2 * i + 1] - mn) / scale), 0, 15);
            qg[i] = (uint8_t)(v0 | (v1 << 4));
            err += std::fabs(v0 * scale + mn - wg[2 * i]) +
                   std::fabs(v1 * scale + mn - wg[2 * i + 1]);
            ref += std::fabs(wg[2 * i]) + std::fabs(wg[2 * i + 1]);
          }
        }
        stat.add(err, ref);
      }
      out[name] = std::move(q);
      out[name + ".q4_scale"] = std::move(sc);
    }
  }

  // ---- write safetensors: 8-byte LE header length, JSON header, data ------
  std::string header = "{";
  int64_t off = 0;
  for (auto& [name, t] : out) {
    header += "\"" + json_escape(name) + "\":{\"dtype\":\"" + t.dtype +
              "\",\"shape\":[";
    for (size_t i = 0; i < t.shape.size(); i++)
      header += (i ? "," : "") + std::to_string(t.shape[i]);
    header += "],\"data_offsets\":[" + std::to_string(off) + "," +
              std::to_string(off + (int64_t)t.bytes.size()) + "]},";
    off += (int64_t)t.bytes.size();
  }
  header.back() = '}';
  while (header.size() % 8 != 0) header += ' ';   // spec: pad header with spaces

  std::ofstream f(argv[2], std::ios::binary);
  if (!f) { fprintf(stderr, "cannot write %s\n", argv[2]); return 1; }
  uint64_t hlen = header.size();
  f.write(reinterpret_cast<const char*>(&hlen), 8);
  f.write(header.data(), (std::streamsize)hlen);
  for (auto& [name, t] : out)
    f.write(reinterpret_cast<const char*>(t.bytes.data()),
            (std::streamsize)t.bytes.size());
  f.close();

  printf("converted %d block projections to %s, rest kept fp32\n", n_quant,
         rt::wtype_name(wt));
  if (wt != rt::WType::F16)
    printf("worst per-row mean relative error: %.4f\n", stat.worst);
  printf("wrote %s (%.1f MB)\n", argv[2], (8.0 + hlen + off) / 1e6);
  return 0;
}
