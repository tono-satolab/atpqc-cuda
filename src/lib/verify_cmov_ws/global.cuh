//
// global.cuh
// Kernel header of verifing ciphertext and constant-time copy.
//
// Copyright (c) 2021 Tatsuki Ono
//
// This software is released under the MIT License.
// https://opensource.org/licenses/mit-license.php
//

#ifndef ATPQC_CUDA_LIB_VERIFY_CMOV_WS_GLOBAL_CUH_
#define ATPQC_CUDA_LIB_VERIFY_CMOV_WS_GLOBAL_CUH_

#include <cstddef>
#include <cstdint>

namespace atpqc_cuda::verify_cmov_ws::global {

__global__ void verify_cmov(std::uint8_t* r, std::size_t r_pitch,
                            const std::uint8_t* x, std::size_t x_pitch,
                            std::size_t move_len, const std::uint8_t* a,
                            std::size_t a_pitch, const std::uint8_t* b,
                            std::size_t b_pitch, std::size_t cmp_len,
                            unsigned ninputs);

}

#endif
