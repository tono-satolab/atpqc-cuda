//
// variants.cuh
// Three variants of Kyber.
//
// Copyright (c) 2021 Tatsuki Ono
//
// This software is released under the MIT License.
// https://opensource.org/licenses/mit-license.php
//

#ifndef ATPQC_CUDA_LIB_KYBER_VARIANTS_CUH_
#define ATPQC_CUDA_LIB_KYBER_VARIANTS_CUH_

namespace atpqc_cuda::kyber::variants {

class kyber512 {};
class kyber768 {};
class kyber1024 {};

inline constexpr kyber512 kyber512_v;
inline constexpr kyber768 kyber768_v;
inline constexpr kyber1024 kyber1024_v;

}  // namespace atpqc_cuda::kyber::variants

#endif
