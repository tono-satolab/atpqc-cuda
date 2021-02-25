//
// zero.cu
// RNG on host generating zeros.
//
// Copyright (c) 2021 Tatsuki Ono
//
// This software is released under the MIT License.
// https://opensource.org/licenses/mit-license.php
//

#ifndef ATPQC_CUDA_LIB_RNG_ZERO_HPP_
#define ATPQC_CUDA_LIB_RNG_ZERO_HPP_

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace atpqc_cuda::rng {

class zero {
 public:
  void operator()(std::uint8_t* x, std::size_t xlen) const {
    std::fill_n(x, xlen, 0x00);
  }
};

}  // namespace atpqc_cuda::rng

#endif
