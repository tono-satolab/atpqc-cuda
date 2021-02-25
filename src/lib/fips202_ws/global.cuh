//
// global.cuh
// SHA-3 kerel header.
//
// Copyright (c) 2021 Tatsuki Ono
//
// This software is released under the MIT License.
// https://opensource.org/licenses/mit-license.php
//

#ifndef ATPQC_CUDA_LIB_FIPS202_WS_GLOBAL_CUH_
#define ATPQC_CUDA_LIB_FIPS202_WS_GLOBAL_CUH_

#include <cstddef>
#include <cstdint>

namespace atpqc_cuda::fips202_ws::global {

template <unsigned Bits>
__global__ void sha3(std::uint8_t* h, std::size_t h_pitch,
                     const std::uint8_t* in, std::size_t in_pitch,
                     std::size_t inlen, unsigned ninputs);

template <unsigned Bits>
__global__ void shake(std::uint8_t* out, std::size_t out_pitch,
                      std::size_t outlen, const std::uint8_t* in,
                      std::size_t in_pitch, std::size_t inlen,
                      unsigned ninputs);

}  // namespace atpqc_cuda::fips202_ws::global

#endif
