//
// device.cuh
// Device function wrapper for XOF and PRF
//
// Copyright (c) 2021 Tatsuki Ono
//
// This software is released under the MIT License.
// https://opensource.org/licenses/mit-license.php
//

#ifndef ATPQC_CUDA_LIB_KYBER_SYMMETRIC_WS_DEVICE_CUH_
#define ATPQC_CUDA_LIB_KYBER_SYMMETRIC_WS_DEVICE_CUH_

#include <cstdint>

#include "../../fips202_ws/device.cuh"
#include "../../fips202_ws/params.cuh"
#include "../params.cuh"

namespace atpqc_cuda::kyber::symmetric_ws::device {

using keccak_type = fips202_ws::device::keccak;
using state_type = keccak_type::state_type;

class xof {
 public:
  static constexpr unsigned blockbytes = fips202_ws::params::shake<128>::rate;

  [[nodiscard]] __device__ state_type absorb(const std::uint8_t* extseed,
                                             const keccak_type& f) const {
    constexpr unsigned extseed_bytes = params::symbytes + 2;
    fips202_ws::device::shake<128>::absorb shake128_absorb;

    return shake128_absorb(extseed, extseed_bytes, f);
  }

  [[nodiscard]] __device__ state_type
  squeezeblocks(std::uint8_t* out, std::size_t outblocks, state_type state,
                const keccak_type& f) const {
    fips202_ws::device::shake<128>::squeezeblocks shake128_squeezeblocks;

    return shake128_squeezeblocks(out, outblocks, state, f);
  }
};

using prf = fips202_ws::device::notmp_shake<256>;

}  // namespace atpqc_cuda::kyber::symmetric_ws::device

#endif
