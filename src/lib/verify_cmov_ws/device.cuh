//
// device.cu
// Device functions for verifing ciphertext and constant-time copy.
//
// Copyright (c) 2021 Tatsuki Ono
//
// This software is released under the MIT License.
// https://opensource.org/licenses/mit-license.php
//

#ifndef ATPQC_CUDA_LIB_VERIFY_CMOV_WS_DEVICE_CUH_
#define ATPQC_CUDA_LIB_VERIFY_CMOV_WS_DEVICE_CUH_

#include <cstddef>
#include <cstdint>

namespace atpqc_cuda::verify_cmov_ws::device {

class verify_cmov {
 public:
  __device__ int verify(const std::uint8_t* a, const std::uint8_t* b,
                        std::size_t len) {
    std::uint32_t r = 0;
    for (unsigned i = 0; i < len; i += 128) {
      unsigned pos = i + (threadIdx.x << 2);
      if (pos < len) r |= a[pos] ^ b[pos];
      ++pos;
      if (pos < len) r |= a[pos] ^ b[pos];
      ++pos;
      if (pos < len) r |= a[pos] ^ b[pos];
      ++pos;
      if (pos < len) r |= a[pos] ^ b[pos];
    }

    r |= __shfl_xor_sync(0xffffffff, r, 0b10000);
    r |= __shfl_xor_sync(0xffffffff, r, 0b01000);
    r |= __shfl_xor_sync(0xffffffff, r, 0b00100);
    r |= __shfl_xor_sync(0xffffffff, r, 0b00010);
    r |= __shfl_xor_sync(0xffffffff, r, 0b00001);

    return (-r) >> 31;
  }

  __device__ void cmov(std::uint8_t* r, const std::uint8_t* x, std::size_t len,
                       std::uint8_t b) {
    b = -b;
    for (unsigned i = 0; i < len; i += 128) {
      unsigned pos = i + (threadIdx.x << 2);
      if (pos < len) r[pos] ^= b & (r[pos] ^ x[pos]);
      ++pos;
      if (pos < len) r[pos] ^= b & (r[pos] ^ x[pos]);
      ++pos;
      if (pos < len) r[pos] ^= b & (r[pos] ^ x[pos]);
      ++pos;
      if (pos < len) r[pos] ^= b & (r[pos] ^ x[pos]);
    }
  }
};

}  // namespace atpqc_cuda::verify_cmov_ws::device

#endif
