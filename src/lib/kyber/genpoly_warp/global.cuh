//
// global.cuh
// Kernel header of generation polynomials.
//
// Copyright (c) 2021 Tatsuki Ono
//
// This software is released under the MIT License.
// https://opensource.org/licenses/mit-license.php
//

#ifndef ATPQC_CUDA_LIB_KYBER_GENPOLY_WARP_GLOBAL_CUH_
#define ATPQC_CUDA_LIB_KYBER_GENPOLY_WARP_GLOBAL_CUH_

#include <cstddef>
#include <cstdint>

namespace atpqc_cuda::kyber::genpoly_warp::global {

template <unsigned K, bool Transposed>
__global__ void genmatrix(short2* polymat, const std::uint8_t* seed,
                          std::size_t seed_pitch, unsigned npolys);

template <unsigned K, unsigned Eta>
__global__ void gennoise(short2* poly, const std::uint8_t* seed,
                         std::size_t seed_pitch, unsigned npolys,
                         std::uint8_t nonce_begin);

}  // namespace atpqc_cuda::kyber::genpoly_warp::global

#endif
