/* stdrng.hpp — the rand 0.9.1 StdRng (ChaCha12) stream.
 *
 * The reference sampler's observable ordering depends on this exact stream,
 * including seed_from_u64's PCG expansion and Canon integer sampling, so any
 * code that has to reproduce a context byte-for-byte must draw from this one
 * implementation rather than a lookalike. Extracted from rt_c.cpp when the
 * graph module needed the same stream.
 */
#ifndef RELATIVEDB_STDRNG_HPP
#define RELATIVEDB_STDRNG_HPP

#include <array>
#include <cstdint>

namespace relrng {

class StdRng091 {
 public:
  explicit StdRng091(uint64_t seed) {
    uint64_t state = seed;
    for (int i = 0; i < 8; ++i) {
      state = state * 6364136223846793005ULL + 11634580027462260723ULL;
      const uint32_t x = (uint32_t)((((state >> 18) ^ state) >> 27));
      const uint32_t rot = (uint32_t)(state >> 59);
      key_[i] = (x >> rot) | (x << ((-rot) & 31));
    }
  }

  uint32_t u32() {
    if (at_ == 64) refill();
    return buf_[at_++];
  }
  uint64_t u64() { return (uint64_t)u32() | ((uint64_t)u32() << 32); }
  uint32_t range(uint32_t stop) {
    const uint64_t product = (uint64_t)u32() * stop;
    uint32_t result = (uint32_t)(product >> 32);
    const uint32_t low = (uint32_t)product;
    if (low > (uint32_t)(-stop)) {
      const uint32_t new_hi = (uint32_t)(((uint64_t)u32() * stop) >> 32);
      if ((uint64_t)low + new_hi > 0xffffffffULL) ++result;
    }
    return result;
  }

 private:
  static uint32_t rotl(uint32_t x, int n) { return (x << n) | (x >> (32 - n)); }
  static void quarter(std::array<uint32_t, 16>& x, int a, int b, int c, int d) {
    x[a] += x[b]; x[d] ^= x[a]; x[d] = rotl(x[d], 16);
    x[c] += x[d]; x[b] ^= x[c]; x[b] = rotl(x[b], 12);
    x[a] += x[b]; x[d] ^= x[a]; x[d] = rotl(x[d], 8);
    x[c] += x[d]; x[b] ^= x[c]; x[b] = rotl(x[b], 7);
  }
  void refill() {
    constexpr uint32_t constants[4] = {
        0x61707865U, 0x3320646eU, 0x79622d32U, 0x6b206574U};
    for (int block = 0; block < 4; ++block) {
      const uint64_t counter = counter_ + block;
      std::array<uint32_t, 16> initial{};
      for (int i = 0; i < 4; ++i) initial[i] = constants[i];
      for (int i = 0; i < 8; ++i) initial[4 + i] = key_[i];
      initial[12] = (uint32_t)counter;
      initial[13] = (uint32_t)(counter >> 32);
      auto x = initial;
      for (int round = 0; round < 6; ++round) {
        quarter(x,0,4,8,12); quarter(x,1,5,9,13);
        quarter(x,2,6,10,14); quarter(x,3,7,11,15);
        quarter(x,0,5,10,15); quarter(x,1,6,11,12);
        quarter(x,2,7,8,13); quarter(x,3,4,9,14);
      }
      for (int i = 0; i < 16; ++i) buf_[block * 16 + i] = x[i] + initial[i];
    }
    counter_ += 4;
    at_ = 0;
  }
  std::array<uint32_t, 8> key_{};
  std::array<uint32_t, 64> buf_{};
  uint64_t counter_ = 0;
  int at_ = 64;
};

}  // namespace relrng

#endif  // RELATIVEDB_STDRNG_HPP
