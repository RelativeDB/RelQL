// Shared Metal buffer allocation.
//
// Metal returns nil when it cannot allocate. Nothing about that is loud:
// binding nil to a compute encoder is legal, so the kernel runs against
// nothing, and the first symptom is a wrong number -- a NaN loss, or a score
// that is simply incorrect -- somewhere downstream with no allocation anywhere
// in the traceback. Every allocation in the engine goes through here so the
// failure names itself at the point it happens.
#pragma once

#import <Metal/Metal.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace rt {
namespace detail {

[[noreturn]] inline void metal_alloc_failed(id<MTLDevice> d, size_t bytes,
                                            const char* what) {
  throw std::runtime_error(
      std::string(what) + ": Metal could not allocate " +
      std::to_string(bytes) + " bytes (" +
      std::to_string((unsigned long long)d.currentAllocatedSize) +
      " already allocated on " + (d.name ? d.name.UTF8String : "?") +
      ", recommended working set " +
      std::to_string((unsigned long long)d.recommendedMaxWorkingSetSize) +
      " bytes)");
}

// Zero-length buffers are not useful to Metal and some paths ask for one when
// a dimension is empty; round up rather than making every caller special-case
// it, which is what the callers were already doing individually.
inline id<MTLBuffer> metal_buffer(id<MTLDevice> d, size_t bytes,
                                  const char* what) {
  id<MTLBuffer> b = [d newBufferWithLength:std::max<size_t>(bytes, 4)
                                   options:MTLResourceStorageModeShared];
  if (!b) metal_alloc_failed(d, bytes, what);
  return b;
}

inline id<MTLBuffer> metal_buffer(id<MTLDevice> d, const void* p, size_t bytes,
                                  const char* what) {
  if (!p || !bytes) return metal_buffer(d, bytes, what);
  id<MTLBuffer> b = [d newBufferWithBytes:p
                                   length:std::max<size_t>(bytes, 4)
                                  options:MTLResourceStorageModeShared];
  if (!b) metal_alloc_failed(d, bytes, what);
  return b;
}

}  // namespace detail
}  // namespace rt
