/* rt_c.h — C ABI for the golden-verified RT-J inference engine (rt.cpp).
 *
 * The shared backend for every relativedb language binding (Rust, Python, Java).
 * Inputs are the raw, PRE-sort token arrays (the engine sorts and builds its
 * sparse attention masks internally — callers never construct masks).
 *
 * BOTH RT-J variants load through the same functions — the architecture is
 * identical; only the weights differ:
 *   classification/model.safetensors -> out_target_scores are LOGITS
 *       (apply sigmoid for probability; bool_as_num routing)
 *   regression/model.safetensors     -> out_target_scores are NORMALIZED
 *       regression values (caller denormalizes with train-split stats)
 * Bindings route per task type: clf/ranking -> classification checkpoint,
 * regression/forecasting -> regression checkpoint (ModelConfig.modelUriFor).
 * Verified against PyTorch on the golden batch: clf max|d|=3.9e-3,
 * reg max|d|=1.0e-3.
 *
 * All arrays are caller-owned, little-endian, densely packed:
 *   length B*S:      node_idxs, col_idxs, table_idxs, sem_types (int64),
 *                    is_padding, is_target (uint8),
 *                    number_v, datetime_v, boolean_v (float32)
 *   length B*S*5:    f2p (int64, -1 = no parent)
 *   length B*S*384:  text_v, col_name_v (float32; MiniLM-L12-v2 embeddings)
 *
 * Thread-safe: one rt_model may be shared across threads; rt_forward is
 * reentrant (all state is per-call).
 */
#ifndef RT_C_H
#define RT_C_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct rt_model rt_model;
typedef struct rt_finetune_head rt_finetune_head;

/* Load a safetensors checkpoint (bf16 or f32). Returns NULL on failure and
 * writes a message into err (if err non-NULL, capped at errlen). */
rt_model* rt_model_load(const char* safetensors_path, char* err, size_t errlen);

void rt_model_free(rt_model*);

/* Number of parameters (for diagnostics). */
int64_t rt_model_num_params(const rt_model*);

/* Compute devices. CPU is always available; MPS/CUDA require the backend to
 * be compiled in (macOS / -DRT_CUDA=ON builds) and a usable device. */
#define RT_DEVICE_CPU 0
#define RT_DEVICE_MPS 1
#define RT_DEVICE_CUDA 2

/* 1 if the device can run rt_forward_device, else 0. */
int rt_device_available(int32_t device);

/* Reference graph-walk sampler primitive.  ``offsets``/``neighbors`` are a
 * CSR graph with deterministic neighbor order; ``eligible`` marks nodes whose
 * visits are counted. The RNG and integer range sampling match rand 0.9.1
 * StdRng. This keeps the product traversal exact while moving the 10k x 20
 * inner loop out of Python. Returns 0 on success. */
int rt_reference_walk_counts(int32_t node_count, const int32_t* offsets,
                             const int32_t* neighbors, int32_t target,
                             const uint8_t* eligible, uint64_t seed,
                             int32_t num_walks, int32_t walk_length,
                             uint32_t* out_counts);

/* First u64 from independently seeded rand 0.9.1 StdRng streams. */
int rt_stdrng_first_u64_batch(const uint64_t* seeds, int32_t count,
                              uint64_t* out_values);

/* Run the forward pass.
 *
 * out_target_scores: length B. For each batch row, the number-head output
 * summed over that row's target positions (each row is expected to carry
 * exactly one target cell — the masked label; rows with none yield 0).
 *
 * n_threads <= 0 selects hardware concurrency.
 * Returns 0 on success, nonzero on error (message in err). */
int rt_forward(const rt_model*, int32_t B, int32_t S,
               const int64_t* node_idxs, const int64_t* f2p,
               const int64_t* col_idxs, const int64_t* table_idxs,
               const uint8_t* is_padding, const int64_t* sem_types,
               const uint8_t* is_target, const float* number_v,
               const float* datetime_v, const float* boolean_v,
               const float* text_v, const float* col_name_v,
               int32_t n_threads, float* out_target_scores,
               char* err, size_t errlen);

/* Extended forward: adds the TEXT decoder head output at the target cell.
 *
 * Identical to rt_forward in every respect (CPU device; out_target_scores is
 * bit-for-bit the same number-head output), with one extra trailing output:
 *
 *   out_target_text: length B*384, or NULL. For each batch row, the dec_dict.text
 *     head output (384-d predicted MiniLM-L12-v2 embedding) summed over that
 *     row's target positions — the SAME target-cell selection rt_forward uses
 *     for the number head. Rows with no target yield all-zeros. The head applies
 *     the model's norm_out RMSNorm then the text Linear, exactly as the number
 *     head does. NOT L2-normalized — the caller normalizes before matching.
 *
 * When out_target_text is NULL this behaves byte-identically to rt_forward.
 * Returns 0 on success, nonzero on error (message in err). */
int rt_forward_ex(const rt_model*, int32_t B, int32_t S,
                  const int64_t* node_idxs, const int64_t* f2p,
                  const int64_t* col_idxs, const int64_t* table_idxs,
                  const uint8_t* is_padding, const int64_t* sem_types,
                  const uint8_t* is_target, const float* number_v,
                  const float* datetime_v, const float* boolean_v,
                  const float* text_v, const float* col_name_v,
                  int32_t n_threads, float* out_target_scores,
                  float* out_target_text, char* err, size_t errlen);

/* Same as rt_forward but on an explicit device (RT_DEVICE_*). n_threads only
 * affects the CPU device. GPU forwards on one model are serialized
 * internally; CPU forwards remain fully reentrant. */
int rt_forward_device(const rt_model*, int32_t B, int32_t S,
                      const int64_t* node_idxs, const int64_t* f2p,
                      const int64_t* col_idxs, const int64_t* table_idxs,
                      const uint8_t* is_padding, const int64_t* sem_types,
                      const uint8_t* is_target, const float* number_v,
                      const float* datetime_v, const float* boolean_v,
                      const float* text_v, const float* col_name_v,
                      int32_t n_threads, int32_t device,
                      float* out_target_scores, char* err, size_t errlen);

/* ---- frozen-backbone fine-tuning --------------------------------------
 *
 * The transformer is used as a frozen relational feature extractor. This
 * call returns its final output-normalized target-cell state [B,512] on the
 * selected device. The compact head API below trains on those states with
 * AdamW on Metal and saves a small safetensors adapter checkpoint.
 */
int rt_encode_targets_device(const rt_model*, int32_t B, int32_t S,
                      const int64_t* node_idxs, const int64_t* f2p,
                      const int64_t* col_idxs, const int64_t* table_idxs,
                      const uint8_t* is_padding, const int64_t* sem_types,
                      const uint8_t* is_target, const float* number_v,
                      const float* datetime_v, const float* boolean_v,
                      const float* text_v, const float* col_name_v,
                      int32_t n_threads, int32_t device,
                      float* out_target_features,
                      char* err, size_t errlen);

#define RT_FINETUNE_BINARY 0
#define RT_FINETUNE_REGRESSION 1
#define RT_FINETUNE_MULTICLASS 2
#define RT_FINETUNE_RANKING 3

/* Create a task head initialized from the released checkpoint. Scalar tasks
 * copy dec_dict.number. For multiclass, class_embeddings may point to
 * n_outputs*384 label embeddings; their projection through dec_dict.text
 * preserves the checkpoint's zero-shot class ordering. NULL initializes a
 * zero multiclass head. */
rt_finetune_head* rt_finetune_head_create(const rt_model*, int32_t task,
                                           int32_t n_outputs,
                                           const float* class_embeddings,
                                           char* err, size_t errlen);
rt_finetune_head* rt_finetune_head_load(const char* path,
                                         char* err, size_t errlen);
void rt_finetune_head_free(rt_finetune_head*);
int rt_finetune_head_save(const rt_finetune_head*, const char* path,
                          char* err, size_t errlen);

/* Preserve the initialized head's raw-feature predictions when subsequent
 * fitting/prediction supplies z=(x-mean)/std features. */
int rt_finetune_head_reparameterize_standardized(
    rt_finetune_head*, const float* mean, const float* std,
    char* err, size_t errlen);

/* Full-batch Metal training over frozen features [N,512]. labels is length N.
 * Ranking uses non-negative relevance labels and group_offsets[n_groups+1];
 * other tasks ignore group_offsets/n_groups. */
int rt_finetune_head_fit_metal(rt_finetune_head*, int32_t N,
                               const float* features, const float* labels,
                               const int32_t* group_offsets, int32_t n_groups,
                               int32_t epochs, float learning_rate,
                               float weight_decay,
                               float* out_initial_loss, float* out_final_loss,
                               double* out_seconds,
                               char* err, size_t errlen);

/* Raw logits/scores [N,n_outputs], evaluated on CPU for portable inference. */
int rt_finetune_head_predict(const rt_finetune_head*, int32_t N,
                             const float* features, float* out_logits,
                             char* err, size_t errlen);
int32_t rt_finetune_head_outputs(const rt_finetune_head*);
int32_t rt_finetune_head_task(const rt_finetune_head*);

/* ---- full-checkpoint MPS fine-tuning ----------------------------------
 * Executes one optimizer step through the complete RT-J model. Target-cell
 * values are labels; is_target prevents them entering the encoder and selects
 * the reference Huber objective. The optimizer state persists on rt_model.
 * This path is native C++/Metal and does not use Python autograd.
 */
int rt_model_finetune_step_metal(
    rt_model*, int32_t B, int32_t S,
    const int64_t* node_idxs, const int64_t* f2p,
    const int64_t* col_idxs, const int64_t* table_idxs,
    const uint8_t* is_padding, const int64_t* sem_types,
    const uint8_t* is_target, const float* number_v,
    const float* datetime_v, const float* boolean_v,
    const float* text_v, const float* col_name_v,
    float learning_rate, float weight_decay, float grad_clip_norm,
    float* out_loss, float* out_grad_norm, uint64_t* out_step,
    double* out_seconds, char* err, size_t errlen);

/* Accumulating variant. apply_update=0 retains gradients without changing
 * weights; apply_update=1 averages all retained microbatches and applies one
 * clipped AdamW update. */
int rt_model_finetune_microbatch_metal(
    rt_model*, int32_t B, int32_t S,
    const int64_t* node_idxs, const int64_t* f2p,
    const int64_t* col_idxs, const int64_t* table_idxs,
    const uint8_t* is_padding, const int64_t* sem_types,
    const uint8_t* is_target, const float* number_v,
    const float* datetime_v, const float* boolean_v,
    const float* text_v, const float* col_name_v,
    float learning_rate, float weight_decay, float grad_clip_norm,
    int32_t apply_update,
    float* out_loss, float* out_grad_norm, uint64_t* out_step,
    uint32_t* out_accumulated_microbatches, int32_t* out_updated,
    double* out_seconds, char* err, size_t errlen);

int rt_model_finetune_optimizer_save(rt_model*, const char* path,
                                     char* err, size_t errlen);
int rt_model_finetune_optimizer_load(rt_model*, const char* path,
                                     char* err, size_t errlen);

int rt_model_finetune_gradient_check_metal(
    rt_model*, int32_t B, int32_t S,
    const int64_t* node_idxs, const int64_t* f2p,
    const int64_t* col_idxs, const int64_t* table_idxs,
    const uint8_t* is_padding, const int64_t* sem_types,
    const uint8_t* is_target, const float* number_v,
    const float* datetime_v, const float* boolean_v,
    const float* text_v, const float* col_name_v,
    float epsilon, float* out_max_absolute_error,
    float* out_max_relative_error, int32_t* out_checked,
    char* err, size_t errlen);

int rt_model_save(const rt_model*, const char* path, char* err, size_t errlen);
void rt_model_finetune_reset_optimizer(rt_model*);

#ifdef __cplusplus
}
#endif
#endif /* RT_C_H */
