//
// global.cuh
// Kernels for encoding/decoding polynomials.
//
// Copyright (c) 2021 Tatsuki Ono
//
// This software is released under the MIT License.
// https://opensource.org/licenses/mit-license.php
//

#ifndef ATPQC_CUDA_LIB_KYBER_ENDECODE_MT_GLOBAL_CUH_
#define ATPQC_CUDA_LIB_KYBER_ENDECODE_MT_GLOBAL_CUH_

#include <cstddef>
#include <cstdint>

namespace atpqc_cuda::kyber::endecode_mt::global {

template <unsigned Dv>
__global__ void poly_compress(std::uint8_t* cbytes, std::size_t cbytes_pitch,
                              const short2* poly, unsigned ninputs);

template <unsigned Dv>
__global__ void poly_decompress(short2* poly, const std::uint8_t* cbytes,
                                std::size_t cbytes_pitch, unsigned ninputs);

template <unsigned K, unsigned Du>
__global__ void polyvec_compress(std::uint8_t* cbytes, std::size_t cbytes_pitch,
                                 const short2* polyvec, unsigned ninputs);

template <unsigned K, unsigned Du>
__global__ void polyvec_decompress(short2* polyvec, const std::uint8_t* cbytes,
                                   std::size_t cbytes_pitch, unsigned ninputs);

template <unsigned K>
__global__ void polyvec_tobytes(std::uint8_t* bytes, std::size_t bytes_pitch,
                                const short2* polyvec, unsigned ninputs);

template <unsigned K>
__global__ void polyvec_frombytes(short2* polyvec, const std::uint8_t* bytes,
                                  std::size_t bytes_pitch, unsigned ninputs);

__global__ void poly_frommsg(short2* poly, const std::uint8_t* msg,
                             std::size_t msg_pitch, unsigned ninputs);

__global__ void poly_tomsg(std::uint8_t* msg, std::size_t msg_pitch,
                           const short2* poly, unsigned ninputs);

}  // namespace atpqc_cuda::kyber::endecode_mt::global

#endif
