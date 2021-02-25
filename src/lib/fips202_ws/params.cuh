//
// params.cuh
// Parameters of SHA-3
//
// Copyright (c) 2021 Tatsuki Ono
//
// This software is released under the MIT License.
// https://opensource.org/licenses/mit-license.php
//

#ifndef ATPQC_CUDA_LIB_FIPS202_WS_PARAMS_CUH_
#define ATPQC_CUDA_LIB_FIPS202_WS_PARAMS_CUH_

namespace atpqc_cuda::fips202_ws::params {

template <unsigned Bits>
struct sha3;
template <>
struct sha3<224> {
  static constexpr unsigned rate = 144;
  static constexpr unsigned outputbytes = 28;
};
template <>
struct sha3<256> {
  static constexpr unsigned rate = 136;
  static constexpr unsigned outputbytes = 32;
};
template <>
struct sha3<384> {
  static constexpr unsigned rate = 104;
  static constexpr unsigned outputbytes = 48;
};
template <>
struct sha3<512> {
  static constexpr unsigned rate = 72;
  static constexpr unsigned outputbytes = 64;
};

template <unsigned Bits>
struct shake;
template <>
struct shake<128> {
  static constexpr unsigned rate = 168;
};
template <>
struct shake<256> {
  static constexpr unsigned rate = 136;
};

}  // namespace atpqc_cuda::fips202_ws::params

#endif
