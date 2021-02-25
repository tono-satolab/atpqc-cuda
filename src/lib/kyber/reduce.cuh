//
// reduce.cuh
// Montgomery/Barrett reduction.
//
// Copyright (c) 2021 Tatsuki Ono
//
// This software is released under the MIT License.
// https://opensource.org/licenses/mit-license.php
//

#ifndef ATPQC_CUDA_LIB_KYBER_REDUCE_CUH_
#define ATPQC_CUDA_LIB_KYBER_REDUCE_CUH_

#include <cstdint>

#include "params.cuh"

namespace atpqc_cuda::kyber::reduce {

inline constexpr unsigned mont_r = 2285;
inline constexpr unsigned q_inv = 62209;

__device__ inline std::int16_t montgomery_reduce(std::int32_t a) noexcept {
  std::int16_t u = a * q_inv;
  std::int32_t t = static_cast<std::int32_t>(u) * params::q;
  t = a - t;
  t >>= 16;
  return t;
}

__device__ inline std::int16_t barrett_reduce(std::int16_t a) noexcept {
  constexpr std::int16_t v = ((1U << 26) + params::q / 2) / params::q;

  std::int16_t t = (static_cast<std::int32_t>(v) * a + (1 << 25)) >> 26;
  t *= params::q;
  return a - t;
}

}  // namespace atpqc_cuda::kyber::reduce

#endif
