//
// global.cuh
// Kernel header of module arithmetic.
//
// Copyright (c) 2021 Tatsuki Ono
//
// This software is released under the MIT License.
// https://opensource.org/licenses/mit-license.php
//

#ifndef ATPQC_CUDA_LIB_KYBER_ARITHMETIC_MT_GLOBAL_CUH_
#define ATPQC_CUDA_LIB_KYBER_ARITHMETIC_MT_GLOBAL_CUH_

namespace atpqc_cuda::kyber::arithmetic_mt::global {

template <unsigned K>
__global__ void mattimesvec_tomont_plusvec(short2* vec_out, const short2* mat,
                                           const short2* vec1,
                                           const short2* vec2);

template <unsigned K>
__global__ void mattimesvec(short2* vec_out, const short2* mat,
                            const short2* vec);

template <unsigned K>
__global__ void vectimesvec(short2* poly_out, const short2* vec1,
                            const short2* vec2);

__global__ void add2(short2* vec_out, const short2* vec1, const short2* vec2);

__global__ void add3(short2* poly_out, const short2* poly1, const short2* poly2,
                     const short2* poly3);

__global__ void sub(short2* poly_out, const short2* poly1, const short2* poly2);

}  // namespace atpqc_cuda::kyber::arithmetic_mt::global

#endif
