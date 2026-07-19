// rt_train.hpp — task-head fine-tuning over frozen RT-J representations.
//
// The transformer forward remains the golden-verified inference path.  A
// training call extracts its final normalized target-cell representation and
// optimizes a small linear task head on Metal with AdamW.  This makes task
// adaptation cheap (512*C + C trainable parameters) and supports losses the
// published RT-J checkpoints never learned: multiclass softmax and listwise
// ranking, in addition to binary classification and regression.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "rt.hpp"

namespace rt {

enum class FineTuneTask : int32_t {
  Binary = 0,
  Regression = 1,
  Multiclass = 2,
  Ranking = 3,
};

struct FineTuneOptions {
  int epochs = 100;
  float learning_rate = 1e-3f;
  float weight_decay = 1e-4f;
  float beta1 = 0.9f;
  float beta2 = 0.999f;
  float epsilon = 1e-8f;
};

struct FineTuneResult {
  float initial_loss = 0.f;
  float final_loss = 0.f;
  int epochs = 0;
  double seconds = 0.0;
};

struct FineTuneHead {
  FineTuneTask task = FineTuneTask::Binary;
  int outputs = 1;
  std::vector<float> weight;  // [outputs, kDModel]
  std::vector<float> bias;    // [outputs]

  // Binary/regression/ranking heads start from the released number decoder.
  // A multiclass head starts from the released text decoder projected onto
  // L2-normalized class-label embeddings [outputs,384].  This preserves the
  // zero-shot class ordering before the first optimizer step.  Pass nullptr
  // to start a multiclass head at zero.
  static FineTuneHead from_model(const Model& model, FineTuneTask task,
                                 int outputs = 1,
                                 const float* class_embeddings = nullptr);

  // Raw logits/scores [N,outputs].  Binary probabilities and multiclass
  // softmax are deliberately left to the caller so ranking can use raw scores.
  std::vector<float> predict(const float* features, int N) const;

  // Small, portable adapter checkpoint (safetensors).
  void save(const std::string& path) const;
  static FineTuneHead load(const std::string& path);
};

// Train the head on Apple Metal. features is [N,512]. labels is length N:
//  - Binary: 0/1
//  - Regression: fp32 target
//  - Multiclass: integer class id encoded as fp32
//  - Ranking: non-negative relevance
// Ranking additionally requires group_offsets [n_groups+1], beginning at 0
// and ending at N; listwise cross-entropy is optimized within each group.
FineTuneResult fit_head_metal(FineTuneHead& head, const float* features,
                              const float* labels, int N,
                              const int32_t* group_offsets, int n_groups,
                              const FineTuneOptions& opts = {});

}  // namespace rt
