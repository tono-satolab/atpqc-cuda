//
// global.cu
// SHA-3 kernel.
//
// Copyright (c) 2021 Tatsuki Ono
//
// This software is released under the MIT License.
// https://opensource.org/licenses/mit-license.php
//

#include "device.cuh"
#include "global.cuh"
#include "params.cuh"

namespace atpqc_cuda::fips202_ws::global {

template <unsigned Bits>
__global__ void sha3(std::uint8_t* h, std::size_t h_pitch,
                     const std::uint8_t* in, std::size_t in_pitch,
                     std::size_t inlen, unsigned ninputs) {
  constexpr unsigned rate = params::sha3<Bits>::rate;
  device::keccak keccak_func;
  device::sha3<Bits> sha3_func;

  extern __shared__ std::uint8_t tmp_shared[];
  const unsigned inputid = blockIdx.x * blockDim.y + threadIdx.y;

  if (inputid < ninputs) {
    std::uint8_t* tmp_shared_ptr = tmp_shared + rate * threadIdx.y;
    h += h_pitch * inputid;
    in += in_pitch * inputid;

    sha3_func(h, in, inlen, keccak_func, tmp_shared_ptr);
  }
}

template <unsigned Bits>
__global__ void shake(std::uint8_t* out, std::size_t out_pitch,
                      std::size_t outlen, const std::uint8_t* in,
                      std::size_t in_pitch, std::size_t inlen,
                      unsigned ninputs) {
  constexpr unsigned rate = params::shake<Bits>::rate;
  device::keccak keccak_func;
  device::shake<Bits> shake_func;

  extern __shared__ std::uint8_t tmp_shared[];
  const unsigned inputid = blockIdx.x * blockDim.y + threadIdx.y;

  if (inputid < ninputs) {
    std::uint8_t* tmp_shared_ptr = tmp_shared + rate * threadIdx.y;
    out += out_pitch * inputid;
    in += in_pitch * inputid;

    shake_func(out, outlen, in, inlen, keccak_func, tmp_shared_ptr);
  }
}

template __global__ void sha3<224>(std::uint8_t*, std::size_t,
                                   const std::uint8_t*, std::size_t,
                                   std::size_t, unsigned);
template __global__ void sha3<256>(std::uint8_t*, std::size_t,
                                   const std::uint8_t*, std::size_t,
                                   std::size_t, unsigned);
template __global__ void sha3<384>(std::uint8_t*, std::size_t,
                                   const std::uint8_t*, std::size_t,
                                   std::size_t, unsigned);
template __global__ void sha3<512>(std::uint8_t*, std::size_t,
                                   const std::uint8_t*, std::size_t,
                                   std::size_t, unsigned);
template __global__ void shake<128>(std::uint8_t*, std::size_t, std::size_t,
                                    const std::uint8_t*, std::size_t,
                                    std::size_t, unsigned);
template __global__ void shake<256>(std::uint8_t*, std::size_t, std::size_t,
                                    const std::uint8_t*, std::size_t,
                                    std::size_t, unsigned);

}  // namespace atpqc_cuda::fips202_ws::global
