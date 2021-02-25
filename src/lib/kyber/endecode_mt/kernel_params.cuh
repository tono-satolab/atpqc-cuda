//
// kernel_params.cuh
// Parameters of kernels for encoding/decoding polynomials.
//
// Copyright (c) 2021 Tatsuki Ono
//
// This software is released under the MIT License.
// https://opensource.org/licenses/mit-license.php
//

#ifndef ATPQC_CUDA_LIB_KYBER_ENDECODE_MT_KERNEL_PARAMS_CUH_
#define ATPQC_CUDA_LIB_KYBER_ENDECODE_MT_KERNEL_PARAMS_CUH_

namespace atpqc_cuda::kyber::endecode_mt::kernel_params {

template <unsigned Dv>
struct poly_de_compress;
template <>
struct poly_de_compress<4> {
  static constexpr unsigned thread_per_input = 128;
  static constexpr unsigned coeff_per_thread = 1;
  static constexpr unsigned cbyte_per_thread = 1;
};
template <>
struct poly_de_compress<5> {
  static constexpr unsigned thread_per_input = 32;
  static constexpr unsigned coeff_per_thread = 4;
  static constexpr unsigned cbyte_per_thread = 5;
};

template <unsigned K, unsigned Du>
struct polyvec_de_compress;
template <unsigned K>
struct polyvec_de_compress<K, 10> {
  static constexpr unsigned thread_per_input = 64 * K;
  static constexpr unsigned coeff_per_thread = 2;
  static constexpr unsigned cbyte_per_thread = 5;
};
template <unsigned K>
struct polyvec_de_compress<K, 11> {
  static constexpr unsigned thread_per_input = 32 * K;
  static constexpr unsigned coeff_per_thread = 4;
  static constexpr unsigned cbyte_per_thread = 11;
};

template <unsigned K>
struct polyvec_bytes {
  static constexpr unsigned thread_per_input = 128 * K;
  static constexpr unsigned coeff_per_thread = 1;
  static constexpr unsigned byte_per_thread = 3;
};

struct poly_msg {
  static constexpr unsigned thread_per_input = 32;
  static constexpr unsigned coeff_per_thread = 4;
  static constexpr unsigned msgbyte_per_thread = 1;
};

}  // namespace atpqc_cuda::kyber::endecode_mt::kernel_params

#endif
