//
// zetas_table.cuh
// Header of constant table for NTT.
//
// Copyright (c) 2021 Tatsuki Ono
//
// This software is released under the MIT License.
// https://opensource.org/licenses/mit-license.php
//

#ifndef ATPQC_CUDA_LIB_KYBER_ZETAS_TABLE_CUH_
#define ATPQC_CUDA_LIB_KYBER_ZETAS_TABLE_CUH_

#include <cstdint>

namespace atpqc_cuda::kyber::zetas_table {

__device__ extern const std::int16_t zetas[128];

}  // namespace atpqc_cuda::kyber::zetas_table

#endif
