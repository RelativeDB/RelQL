// Shared Metal shader-compilation options.
//
// Every shader in the engine wants safe math: the kernels rely on exp and
// rsqrt behaving at fp32 precision, and fast math is free to reassociate and
// contract them. Saying so portably takes more than one line.
//
// mathMode/MTLMathModeSafe arrived in the macOS 15 SDK and deprecated
// fastMathEnabled. @available gates the *runtime* only -- the symbol still has
// to exist when the file is compiled -- so an SDK without it fails to build
// even inside an @available branch. That is not hypothetical: CI runs
// macos-latest (15+ SDK) while wheels.yml pins macos-14 (Xcode 15.4), so the
// macOS wheel would not compile while every CI lane was green.
#pragma once

#import <Metal/Metal.h>

namespace rt {
namespace detail {

// Configure `o` for fp32-precise exp/rsqrt on whatever SDK is in use.
inline void set_safe_math(MTLCompileOptions* o) {
#if defined(MAC_OS_VERSION_15_0) && \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= MAC_OS_VERSION_15_0
  if (@available(macOS 15.0, *)) {
    o.mathMode = MTLMathModeSafe;
    return;
  }
#endif
  // Pre-15 SDK, or a 15+ SDK running on an older OS. fastMathEnabled is the
  // equivalent knob and is deprecated rather than gone.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
  o.fastMathEnabled = NO;
#pragma clang diagnostic pop
}

}  // namespace detail
}  // namespace rt
