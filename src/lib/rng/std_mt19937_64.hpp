//
// std_mt19937_64.cu
// RNG on host using Mersenne twister.
//
// Copyright (c) 2021 Tatsuki Ono
//
// This software is released under the MIT License.
// https://opensource.org/licenses/mit-license.php
//

#ifndef ATPQC_CUDA_LIB_RNG_STD_MT19937_64_HPP_
#define ATPQC_CUDA_LIB_RNG_STD_MT19937_64_HPP_

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <random>

namespace atpqc_cuda::rng {

class std_mt19937_64 {
 private:
  static inline auto engine = std::mt19937_64();

  static inline std::mutex mtx;

 public:
  void operator()(std::uint8_t* x, std::size_t xlen) const {
    constexpr unsigned u8t_size = sizeof(std::uint8_t);
    constexpr unsigned rand_size = sizeof(std::mt19937_64::result_type);
    constexpr unsigned n = rand_size / u8t_size;

    auto lock = std::lock_guard(mtx);

    for (const auto x_end = x + xlen; x != x_end;) {
      std::mt19937_64::result_type r = engine();
      for (const auto x_end_inner = x + n; x != x_end_inner && x != x_end;
           ++x) {
        *x = r;
        r >>= 8;
      }
    }
  }

  void seed(std::mt19937_64::result_type value =
                std::mt19937_64::default_seed) const {
    engine.seed(value);
  }
};

}  // namespace atpqc_cuda::rng

#endif
