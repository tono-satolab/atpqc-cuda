//
// host.cuh
// Launches kernel of verifing ciphertext and constant-time copy.
//
// Copyright (c) 2021 Tatsuki Ono
//
// This software is released under the MIT License.
// https://opensource.org/licenses/mit-license.php
//

#ifndef ATPQC_CUDA_LIB_VERIFY_CMOV_WS_HOST_CUH_
#define ATPQC_CUDA_LIB_VERIFY_CMOV_WS_HOST_CUH_

#include <cstddef>
#include <cstdint>
#include <memory>

#include "../cuda_resource.hpp"
#include "global.cuh"

namespace atpqc_cuda::verify_cmov_ws::host {

class verify_cmov {
 private:
  dim3 dg;
  dim3 db;
  unsigned ns;
  unsigned nin;

 public:
  void operator()(std::uint8_t* r, std::size_t r_pitch, const std::uint8_t* x,
                  std::size_t x_pitch, std::size_t move_len,
                  const std::uint8_t* a, std::size_t a_pitch,
                  const std::uint8_t* b, std::size_t b_pitch,
                  std::size_t cmp_len, cudaStream_t stream) const {
    global::verify_cmov<<<dg, db, ns, stream>>>(
        r, r_pitch, x, x_pitch, move_len, a, a_pitch, b, b_pitch, cmp_len, nin);
  }

  explicit verify_cmov(unsigned ninputs, unsigned warp_per_block = 4)
      : dg(make_uint3((ninputs + warp_per_block - 1) / warp_per_block, 1, 1)),
        db(make_uint3(32, warp_per_block, 1)),
        ns(0),
        nin(ninputs) {}

  verify_cmov() = delete;

  void* get_func() const {
    return reinterpret_cast<void*>(global::verify_cmov);
  }
  dim3 get_grid_dim() const { return dg; }
  dim3 get_block_dim() const { return db; }
  unsigned get_shared_bytes() const { return ns; }
  using args_type = cuda_resource::kernel_args<
      std::uint8_t*, std::size_t, const std::uint8_t*, std::size_t, std::size_t,
      const std::uint8_t*, std::size_t, const std::uint8_t*, std::size_t,
      std::size_t, unsigned>;
  std::unique_ptr<args_type> generate_args(
      std::uint8_t* r, std::size_t r_pitch, const std::uint8_t* x,
      std::size_t x_pitch, std::size_t move_len, const std::uint8_t* a,
      std::size_t a_pitch, const std::uint8_t* b, std::size_t b_pitch,
      std::size_t cmp_len) const noexcept {
    return std::make_unique<args_type>(r, r_pitch, x, x_pitch, move_len, a,
                                       a_pitch, b, b_pitch, cmp_len, nin);
  }
};

}  // namespace atpqc_cuda::verify_cmov_ws::host

#endif
