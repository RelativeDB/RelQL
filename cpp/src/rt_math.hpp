// rt_math.hpp — CPU math primitives behind one interface.
//
// Apple builds use Accelerate (cblas sgemm + vDSP + vvexpf). Everywhere else
// (or when RT_PORTABLE is defined) a portable, register-blocked fallback is
// used: the GEMM keeps a 4x8 accumulator tile in registers so -O3 can
// vectorize the inner FMA loop, and the vector ops are plain autovectorizable
// loops. Numerics match Accelerate within normal fp32 reassociation error.
#pragma once

#include <cmath>
#include <cstddef>

#if defined(__APPLE__) && !defined(RT_PORTABLE)
#define RT_ACCELERATE 1
#include <Accelerate/Accelerate.h>
#endif

namespace rt {
namespace math {

#if RT_ACCELERATE

// C[M,N] = A[M,K] @ B[N,K]^T  (row-major, arbitrary leading dims)
inline void gemm_nt(const float* a, const float* b, float* c, int M, int N,
                    int K, int lda, int ldb, int ldc, float beta = 0.0f) {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0f, a, lda,
              b, ldb, beta, c, ldc);
}

// C[M,N] = A[M,K] @ B[K,N]  (row-major, arbitrary leading dims)
inline void gemm_nn(const float* a, const float* b, float* c, int M, int N,
                    int K, int lda, int ldb, int ldc) {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, a, lda,
              b, ldb, 0.0f, c, ldc);
}

inline void vadd(const float* a, const float* b, float* dst, size_t n) {
  vDSP_vadd(a, 1, b, 1, dst, 1, n);
}
inline void vmul(const float* a, const float* b, float* dst, size_t n) {
  vDSP_vmul(a, 1, b, 1, dst, 1, n);
}
// dst = num / den
inline void vdiv(const float* num, const float* den, float* dst, size_t n) {
  vDSP_vdiv(den, 1, num, 1, dst, 1, n);
}
inline void vsmul(const float* a, float s, float* dst, size_t n) {
  vDSP_vsmul(a, 1, &s, dst, 1, n);
}
inline void vsadd(const float* a, float s, float* dst, size_t n) {
  vDSP_vsadd(a, 1, &s, dst, 1, n);
}
inline void vneg(const float* a, float* dst, size_t n) {
  vDSP_vneg(a, 1, dst, 1, n);
}
inline void vexp_inplace(float* a, int n) { vvexpf(a, a, &n); }
inline float maxv(const float* a, size_t n) {
  float m;
  vDSP_maxv(a, 1, &m, n);
  return m;
}
inline float sum(const float* a, size_t n) {
  float s;
  vDSP_sve(a, 1, &s, n);
  return s;
}
inline float dot(const float* a, const float* b, size_t n) {
  float d;
  vDSP_dotpr(a, 1, b, 1, &d, n);
  return d;
}

#else  // ---- portable fallback ---------------------------------------------

// 4x8 register-blocked micro-kernel: C[M,N] = A[M,K] @ B[N,K]^T. Both
// operands walk K contiguously, so the compiler vectorizes the FMA chain and
// the accumulator tile stays in registers across the K loop.
inline void gemm_nt(const float* a, const float* b, float* c, int M, int N,
                    int K, int lda, int ldb, int ldc, float beta = 0.0f) {
  constexpr int MR = 4, NR = 8;
  for (int i0 = 0; i0 < M; i0 += MR) {
    int mi = M - i0 < MR ? M - i0 : MR;
    for (int j0 = 0; j0 < N; j0 += NR) {
      int nj = N - j0 < NR ? N - j0 : NR;
      if (mi == MR && nj == NR) {
        float acc[MR][NR];
        for (int i = 0; i < MR; i++)
          for (int j = 0; j < NR; j++)
            acc[i][j] = beta == 0.f
                            ? 0.f
                            : beta * c[(size_t)(i0 + i) * ldc + j0 + j];
        const float* a0 = a + (size_t)i0 * lda;
        const float* b0 = b + (size_t)j0 * ldb;
        for (int k = 0; k < K; k++) {
          float av[MR];
          for (int i = 0; i < MR; i++) av[i] = a0[(size_t)i * lda + k];
          for (int j = 0; j < NR; j++) {
            float bv = b0[(size_t)j * ldb + k];
            for (int i = 0; i < MR; i++) acc[i][j] += av[i] * bv;
          }
        }
        for (int i = 0; i < MR; i++)
          for (int j = 0; j < NR; j++)
            c[(size_t)(i0 + i) * ldc + j0 + j] = acc[i][j];
      } else {  // edge tile
        for (int i = 0; i < mi; i++)
          for (int j = 0; j < nj; j++) {
            const float* ar = a + (size_t)(i0 + i) * lda;
            const float* br = b + (size_t)(j0 + j) * ldb;
            float s = beta == 0.f
                          ? 0.f
                          : beta * c[(size_t)(i0 + i) * ldc + j0 + j];
            for (int k = 0; k < K; k++) s += ar[k] * br[k];
            c[(size_t)(i0 + i) * ldc + j0 + j] = s;
          }
      }
    }
  }
}

// C[M,N] = A[M,K] @ B[K,N]: accumulate row-panels of B so the j loop
// vectorizes; C row stays hot in cache/registers across the K loop blocks.
inline void gemm_nn(const float* a, const float* b, float* c, int M, int N,
                    int K, int lda, int ldb, int ldc) {
  for (int i = 0; i < M; i++) {
    float* cr = c + (size_t)i * ldc;
    for (int j = 0; j < N; j++) cr[j] = 0.f;
    const float* ar = a + (size_t)i * lda;
    for (int k = 0; k < K; k++) {
      float av = ar[k];
      const float* br = b + (size_t)k * ldb;
      for (int j = 0; j < N; j++) cr[j] += av * br[j];
    }
  }
}

inline void vadd(const float* a, const float* b, float* dst, size_t n) {
  for (size_t i = 0; i < n; i++) dst[i] = a[i] + b[i];
}
inline void vmul(const float* a, const float* b, float* dst, size_t n) {
  for (size_t i = 0; i < n; i++) dst[i] = a[i] * b[i];
}
inline void vdiv(const float* num, const float* den, float* dst, size_t n) {
  for (size_t i = 0; i < n; i++) dst[i] = num[i] / den[i];
}
inline void vsmul(const float* a, float s, float* dst, size_t n) {
  for (size_t i = 0; i < n; i++) dst[i] = a[i] * s;
}
inline void vsadd(const float* a, float s, float* dst, size_t n) {
  for (size_t i = 0; i < n; i++) dst[i] = a[i] + s;
}
inline void vneg(const float* a, float* dst, size_t n) {
  for (size_t i = 0; i < n; i++) dst[i] = -a[i];
}
inline void vexp_inplace(float* a, int n) {
  for (int i = 0; i < n; i++) a[i] = std::exp(a[i]);
}
inline float maxv(const float* a, size_t n) {
  float m = a[0];
  for (size_t i = 1; i < n; i++) m = a[i] > m ? a[i] : m;
  return m;
}
inline float sum(const float* a, size_t n) {
  float s = 0.f;
  for (size_t i = 0; i < n; i++) s += a[i];
  return s;
}
inline float dot(const float* a, const float* b, size_t n) {
  float s = 0.f;
  for (size_t i = 0; i < n; i++) s += a[i] * b[i];
  return s;
}

#endif

}  // namespace math
}  // namespace rt
