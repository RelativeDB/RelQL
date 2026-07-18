#include "csc_c.h"

#include <cstring>
#include <exception>
#include <string>

#include "csc.hpp"

namespace {
void set_str(char* dst, size_t dstlen, const std::string& msg) {
  if (!dst || dstlen == 0) return;
  size_t n = msg.size();
  if (n > dstlen - 1) n = dstlen - 1;
  std::memcpy(dst, msg.data(), n);
  dst[n] = '\0';
}
}  // namespace

struct csc_index {
  csc::CscAdjacency adj;
};

extern "C" {

csc_index* csc_build(int64_t n_parents, int64_t n_edges,
                     const int64_t* edge_parent, const int64_t* edge_child,
                     const double* edge_ts, char* err, size_t errlen) {
  try {
    if (n_edges > 0 &&
        (edge_parent == nullptr || edge_child == nullptr || edge_ts == nullptr)) {
      set_str(err, errlen, "null edge array with n_edges > 0");
      return nullptr;
    }
    return new csc_index{
        csc::CscAdjacency(n_parents, n_edges, edge_parent, edge_child, edge_ts)};
  } catch (const std::exception& e) {
    set_str(err, errlen, e.what());
    return nullptr;
  } catch (...) {
    set_str(err, errlen, "unknown error");
    return nullptr;
  }
}

void csc_free(csc_index* idx) { delete idx; }

int csc_children(const csc_index* idx, int64_t parent_dense, double anchor_ts,
                 int32_t limit, int64_t* out_child, int32_t* out_n,
                 char* err, size_t errlen) {
  try {
    if (idx == nullptr) {
      set_str(err, errlen, "null index");
      return 1;
    }
    if (limit > 0 && out_child == nullptr) {
      set_str(err, errlen, "null out_child with limit > 0");
      return 2;
    }
    int32_t n = idx->adj.children(parent_dense, anchor_ts, limit, out_child);
    if (out_n) *out_n = n;
    return 0;
  } catch (const std::exception& e) {
    set_str(err, errlen, e.what());
    return 1;
  } catch (...) {
    set_str(err, errlen, "unknown error");
    return 1;
  }
}

}  // extern "C"
