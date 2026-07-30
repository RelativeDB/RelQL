// minilm_internal.hpp — weight layout shared between the CPU encoder
// (minilm.cpp) and the CUDA encoder (minilm_cuda.cu).
#ifndef RELATIVEDB_MINILM_INTERNAL_HPP
#define RELATIVEDB_MINILM_INTERNAL_HPP

#include <cstdint>
#include <string>
#include <vector>

namespace minilm {
namespace detail {

struct LayerW {
  const float *q_w, *q_b, *k_w, *k_b, *v_w, *v_b;
  const float *ao_w, *ao_b, *aln_w, *aln_b;    // attention output + LN
  const float *i_w, *i_b;                      // intermediate (FFN up)
  const float *o_w, *o_b, *oln_w, *oln_b;      // FFN down + LN
};

struct HostWeights {
  const float *word_emb, *pos_emb, *type_emb, *eln_w, *eln_b;
  const LayerW* layers;   // [12]
  int64_t vocab_rows = 0;
};

#ifdef RT_CUDA
bool cuda_available();
// Encode pre-tokenized texts on the device; writes unit-norm [n, kDim] rows
// (the pipeline's Normalize module, same as the CPU path). Texts sharing a
// token count batch into strided GEMMs; the math per text is unchanged, only
// fp reduction order differs (absorbed by the downstream bf16 rounding).
// Returns false with *err set on any CUDA failure; the caller falls back.
bool cuda_encode(const HostWeights& w,
                 const std::vector<std::vector<int32_t>>& ids, float* out,
                 std::string* err);
#endif

}  // namespace detail
}  // namespace minilm

#endif  // RELATIVEDB_MINILM_INTERNAL_HPP
