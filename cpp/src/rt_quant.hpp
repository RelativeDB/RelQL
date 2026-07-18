// rt_quant.hpp — quantized weight formats shared by the quantizer, the
// loader and the CPU/MPS compute paths.
//
// All formats quantize along the input dimension of a row-major [out, in]
// weight; rows are independent, so stacked (fused qkvg) weights are plain
// payload concatenation. Activations are always fp32 — dequantization
// happens next to the FMA (CPU: per-tile into L2 scratch; Metal: in-register
// while staging threadgroup tiles).
//
//   F16:  IEEE half payload, no scales.
//   Q8:   int8 per-output-row symmetric, fp32 scale per row.
//         w[o,i] ~= q[o,i] * scale[o],  scale = max|row| / 127
//   Q4:   uint4 in groups of 32 along the input dim (Q4_1-style asymmetric),
//         fp16 (scale, min) pair per group, two values per byte
//         (low nibble = even i):
//         w[o,i] ~= nibble * scale[o, i/32] + min[o, i/32],
//         scale = (max - min) / 15
#pragma once

#include <cstdint>
#include <cstring>

namespace rt {

enum class WType : uint8_t { F32 = 0, F16 = 1, Q8 = 2, Q4 = 3 };

constexpr int kQ4Group = 32;

inline const char* wtype_name(WType t) {
  switch (t) {
    case WType::F32: return "f32";
    case WType::F16: return "f16";
    case WType::Q8: return "q8";
    case WType::Q4: return "q4";
  }
  return "?";
}

// payload bytes for one [in]-long row
inline size_t row_bytes(WType t, int in) {
  switch (t) {
    case WType::F32: return (size_t)in * 4;
    case WType::F16: return (size_t)in * 2;
    case WType::Q8: return (size_t)in;
    case WType::Q4: return (size_t)in / 2;
  }
  return 0;
}

// scale bytes for one row
inline size_t scale_bytes(WType t, int in) {
  switch (t) {
    case WType::Q8: return 4;                        // f32 per row
    case WType::Q4: return (size_t)(in / kQ4Group) * 4;  // f16 (scale,min)/group
    default: return 0;
  }
}

// ---- IEEE half <-> float --------------------------------------------------
#if defined(__FLT16_MAX__)
inline float half_to_float(uint16_t h) {
  _Float16 v;
  std::memcpy(&v, &h, 2);
  return (float)v;
}
inline uint16_t float_to_half(float f) {
  _Float16 v = (_Float16)f;
  uint16_t u;
  std::memcpy(&u, &v, 2);
  return u;
}
#else
inline float half_to_float(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1f;
  uint32_t man = h & 0x3ffu;
  uint32_t u;
  if (exp == 0) {
    if (man == 0) {
      u = sign;
    } else {                       // subnormal: normalize
      exp = 127 - 15 + 1;
      while (!(man & 0x400u)) { man <<= 1; exp--; }
      man &= 0x3ffu;
      u = sign | (exp << 23) | (man << 13);
    }
  } else if (exp == 31) {
    u = sign | 0x7f800000u | (man << 13);
  } else {
    u = sign | ((exp - 15 + 127) << 23) | (man << 13);
  }
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
inline uint16_t float_to_half(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  uint32_t sign = (u >> 16) & 0x8000u;
  int32_t exp = (int32_t)((u >> 23) & 0xff) - 127 + 15;
  uint32_t man = u & 0x7fffffu;
  if (exp >= 31) return (uint16_t)(sign | 0x7c00u);           // inf/overflow
  if (exp <= 0) {                                             // subnormal/0
    if (exp < -10) return (uint16_t)sign;
    man |= 0x800000u;
    uint32_t shift = (uint32_t)(14 - exp);
    uint32_t half_man = man >> shift;
    if ((man >> (shift - 1)) & 1) half_man++;                 // round nearest
    return (uint16_t)(sign | half_man);
  }
  uint16_t h = (uint16_t)(sign | (exp << 10) | (man >> 13));
  if (man & 0x1000u) h++;                                     // round nearest
  return h;
}
#endif

// ---- row dequantization (CPU tile path) -----------------------------------
// q: row payload; qs: row scales; dst: fp32 [in].
inline void dequant_row_f16(const uint8_t* q, float* dst, int in) {
#if defined(__FLT16_MAX__)
  const _Float16* h = reinterpret_cast<const _Float16*>(q);
  for (int i = 0; i < in; i++) dst[i] = (float)h[i];
#else
  const uint16_t* h = reinterpret_cast<const uint16_t*>(q);
  for (int i = 0; i < in; i++) dst[i] = half_to_float(h[i]);
#endif
}

inline void dequant_row_q8(const uint8_t* q, const uint8_t* qs, float* dst,
                           int in) {
  float s;
  std::memcpy(&s, qs, 4);
  const int8_t* p = reinterpret_cast<const int8_t*>(q);
  for (int i = 0; i < in; i++) dst[i] = s * (float)p[i];
}

inline void dequant_row_q4(const uint8_t* q, const uint8_t* qs, float* dst,
                           int in) {
  const uint16_t* sh = reinterpret_cast<const uint16_t*>(qs);
  for (int g = 0; g < in / kQ4Group; g++) {
    const float s = half_to_float(sh[2 * g]);
    const float mn = half_to_float(sh[2 * g + 1]);
    const uint8_t* b = q + g * (kQ4Group / 2);
    float* d = dst + g * kQ4Group;
    for (int i = 0; i < kQ4Group / 2; i++) {
      d[2 * i] = s * (float)(b[i] & 0xf) + mn;
      d[2 * i + 1] = s * (float)(b[i] >> 4) + mn;
    }
  }
}

}  // namespace rt
