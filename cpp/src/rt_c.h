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

#ifdef __cplusplus
}
#endif
#endif /* RT_C_H */
