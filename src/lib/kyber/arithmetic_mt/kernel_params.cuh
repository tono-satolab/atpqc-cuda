//
// kernel_params.cuh
// Parameters for module arithmetic kernels.
//
// Copyright (c) 2021 Tatsuki Ono
//
// This software is released under the MIT License.
// https://opensource.org/licenses/mit-license.php
//

#ifndef ATPQC_CUDA_LIB_KYBER_ARITHMETIC_MT_KERNEL_PARAMS_CUH_
#define ATPQC_CUDA_LIB_KYBER_ARITHMETIC_MT_KERNEL_PARAMS_CUH_

#include <cstdint>

#include "../params.cuh"

namespace atpqc_cuda::kyber::arithmetic_mt::kernel_params {

template <unsigned K>
struct mattimes {
  static constexpr unsigned k = K;
  static constexpr unsigned smem_byte_per_coeff = k * k * sizeof(short2);
};

}  // namespace atpqc_cuda::kyber::arithmetic_mt::kernel_params

#endif
