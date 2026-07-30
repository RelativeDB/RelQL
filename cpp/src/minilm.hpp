/* minilm.hpp — the pinned MiniLM-L12-v2 text encoder, native.
 *
 * RT-J is frozen against vectors this exact encoder produced
 * (sentence-transformers/all-MiniLM-L12-v2: BERT 12 layers x 384 hidden,
 * WordPiece, 128-token window, mean pooling), so whoever runs the
 * transformer must also run the encoder — the serving backend and the
 * in-process engine both reach it here; no Python embedding path exists.
 *
 * Loads the HF snapshot directory (model.safetensors + vocab.txt) directly;
 * resolve_minilm_snapshot() finds one in the HF cache. Verified against
 * sentence-transformers goldens by test_minilm.cpp.
 *
 * Tokenizer parity notes: BERT basic tokenization (lowercase, NFD accent
 * stripping, punctuation splitting, CJK spacing) is implemented over UTF-8
 * with embedded tables covering Latin-1/Latin-Extended-A diacritics, general
 * punctuation and the CJK blocks. Exotic scripts outside those tables pass
 * through un-normalized and typically land on [UNK] — the same bucket a
 * full-Unicode tokenizer sends unknown words to.
 */
#ifndef RELATIVEDB_MINILM_HPP
#define RELATIVEDB_MINILM_HPP

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace minilm {

constexpr int kDim = 384;
constexpr int kMaxTokens = 128;   // the shipped window; see rt kb F13/F14

// Newest HF-cache snapshot of sentence-transformers/all-MiniLM-L12-v2 that
// contains model.safetensors, honoring $HF_HOME; "" when none is found.
std::string resolve_snapshot();

class MiniLM {
 public:
  // snapshot_dir may be empty -> resolve_snapshot(). Returns nullptr on
  // failure with a message in *err.
  static std::unique_ptr<MiniLM> load(const std::string& snapshot_dir,
                                      std::string* err);
  ~MiniLM();

  // texts -> out[n * kDim], mean-pooled; L2-normalized iff normalize.
  // Thread-safe: all per-call state is local.
  bool encode(const std::vector<std::string>& texts, bool normalize,
              float* out, std::string* err) const;

  // Token ids for one text ([CLS] ... [SEP], truncated to kMaxTokens) —
  // exposed for tokenizer-parity tests.
  std::vector<int32_t> tokenize(const std::string& text) const;

 private:
  MiniLM();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace minilm

#endif  // RELATIVEDB_MINILM_HPP
