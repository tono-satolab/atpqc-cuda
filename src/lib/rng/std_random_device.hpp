//
// std_random_device.cu
// RNG on host using random_device of C++.
//
// Copyright (c) 2021 Tatsuki Ono
//
// This software is released under the MIT License.
// https://opensource.org/licenses/mit-license.php
//

#ifndef ATPQC_CUDA_LIB_RNG_STD_RANDOM_DEVICE_HPP_
#define ATPQC_CUDA_LIB_RNG_STD_RANDOM_DEVICE_HPP_

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <random>

namespace atpqc_cuda::rng {

class std_random_device {
 private:
  static inline auto rd = std::random_device();

  static inline std::mutex mtx;

 public:
  void operator()(std::uint8_t* x, std::size_t xlen) const {
    constexpr unsigned u8t_size = sizeof(std::uint8_t);
    constexpr unsigned rand_size = sizeof(std::random_device::result_type);
    constexpr unsigned n = rand_size / u8t_size;

    auto lock = std::lock_guard(mtx);

    for (const auto x_end = x + xlen; x != x_end;) {
      std::random_device::result_type r = rd();
      for (const auto x_end_inner = x + n; x != x_end_inner && x != x_end;
           ++x) {
        *x = r;
        r >>= 8;
      }
    }
  }
};

}  // namespace atpqc_cuda::rng

#endif
