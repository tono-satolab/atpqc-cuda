//
// common.cuh
// Parameter settings of matrices, modules, and polynomials.
//
// Copyright (c) 2021 Tatsuki Ono
//
// This software is released under the MIT License.
// https://opensource.org/licenses/mit-license.php
//

#ifndef ATPQC_CUDA_LIB_KYBER_PRIMITIVE_COMMON_CUH_
#define ATPQC_CUDA_LIB_KYBER_PRIMITIVE_COMMON_CUH_

#include "../params.cuh"

namespace atpqc_cuda::kyber::primitive {

template <class Variant>
struct poly_size {
  using variant = Variant;
  static constexpr unsigned poly = params::n / 2;
  static constexpr unsigned vec = params::k<variant> * poly;
  static constexpr unsigned mat = params::k<variant> * vec;
};

}  // namespace atpqc_cuda::kyber::primitive

#endif
