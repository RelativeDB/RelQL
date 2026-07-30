/* minilm_c.h — C ABI for the native MiniLM text encoder (minilm.hpp).
 *
 * Compiled into librt_c so the Python engine package embeds text through the
 * same encoder the serving backend runs — text embedding never happens in
 * Python. Conventions follow rt_c.h: opaque handle, err buffer, 0 on success.
 */
#ifndef RELATIVEDB_MINILM_C_H
#define RELATIVEDB_MINILM_C_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct minilm_t minilm_t;

/* snapshot_dir may be NULL/empty -> auto-resolve the newest HF-cache snapshot
 * of sentence-transformers/all-MiniLM-L12-v2 (honors $HF_HOME, default
 * ~/.cache/huggingface). Returns NULL on failure, message in err. */
minilm_t* minilm_load(const char* snapshot_dir, char* err, size_t errlen);

void minilm_free(minilm_t*);

/* texts: n UTF-8 NUL-terminated strings. out: n*384 floats (mean-pooled,
 * L2-normalized iff normalize != 0). Thread-safe on one handle. 0 on
 * success, nonzero on error. */
int minilm_encode(const minilm_t*, const char* const* texts, int32_t n,
                  int32_t normalize, float* out, char* err, size_t errlen);

int32_t minilm_dim(const minilm_t*);   /* 384 */

#ifdef __cplusplus
}
#endif

#endif /* RELATIVEDB_MINILM_C_H */
