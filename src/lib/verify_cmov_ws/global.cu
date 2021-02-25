//
// global.cu
// Kernel of verifing ciphertext and constant-time copy.
//
// Copyright (c) 2021 Tatsuki Ono
//
// This software is released under the MIT License.
// https://opensource.org/licenses/mit-license.php
//

#include "device.cuh"
#include "global.cuh"

namespace atpqc_cuda::verify_cmov_ws::global {

__global__ void verify_cmov(std::uint8_t* r, std::size_t r_pitch,
                            const std::uint8_t* x, std::size_t x_pitch,
                            std::size_t cmove_len, const std::uint8_t* a,
                            std::size_t a_pitch, const std::uint8_t* b,
                            std::size_t b_pitch, std::size_t verify_len,
                            unsigned ninputs) {
  device::verify_cmov vc;

  if (unsigned pos = blockIdx.x * blockDim.y + threadIdx.y; pos < ninputs) {
    r += r_pitch * pos;
    x += x_pitch * pos;
    a += a_pitch * pos;
    b += b_pitch * pos;

    vc.cmov(r, x, cmove_len, vc.verify(a, b, verify_len));
  }
}

}  // namespace atpqc_cuda::verify_cmov_ws::global
