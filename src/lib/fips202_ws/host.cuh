//
// host.cuh
// Launches SHA-3 kernel.
//
// Copyright (c) 2021 Tatsuki Ono
//
// This software is released under the MIT License.
// https://opensource.org/licenses/mit-license.php
//

#ifndef ATPQC_CUDA_LIB_FIPS202_WS_HOST_CUH_
#define ATPQC_CUDA_LIB_FIPS202_WS_HOST_CUH_

#include <cstddef>
#include <cstdint>
#include <memory>

#include "../cuda_resource.hpp"
#include "global.cuh"
#include "params.cuh"

namespace atpqc_cuda::fips202_ws::host {

template <unsigned Bits>
class sha3 {
 private:
  dim3 dg;
  dim3 db;
  unsigned ns;
  unsigned nin;

 public:
  static constexpr unsigned rate = params::sha3<Bits>::rate;
  static constexpr unsigned outputbytes = params::sha3<Bits>::outputbytes;

  void operator()(std::uint8_t* h, std::size_t h_pitch, const std::uint8_t* in,
                  std::size_t in_pitch, std::size_t inlen,
                  cudaStream_t stream) const {
    global::sha3<Bits>
        <<<dg, db, ns, stream>>>(h, h_pitch, in, in_pitch, inlen, nin);
  }

  explicit sha3(unsigned ninputs, unsigned warp_per_block)
      : dg(make_uint3((ninputs - 1 + warp_per_block) / warp_per_block, 1, 1)),
        db(make_uint3(32, warp_per_block, 1)),
        ns(rate * warp_per_block),
        nin(ninputs) {}

  sha3() = delete;

  void* get_func() const { return reinterpret_cast<void*>(global::sha3<Bits>); }
  dim3 get_grid_dim() const { return dg; }
  dim3 get_block_dim() const { return db; }
  unsigned get_shared_bytes() const { return ns; }
  using args_type = cuda_resource::kernel_args<std::uint8_t*, std::size_t,
                                               const std::uint8_t*, std::size_t,
                                               std::size_t, unsigned>;
  std::unique_ptr<args_type> generate_args(std::uint8_t* h, std::size_t h_pitch,
                                           const std::uint8_t* in,
                                           std::size_t in_pitch,
                                           std::size_t inlen) const noexcept {
    return std::make_unique<args_type>(h, h_pitch, in, in_pitch, inlen, nin);
  }
};

template <unsigned Bits>
class shake {
 private:
  dim3 dg;
  dim3 db;
  unsigned ns;
  unsigned nin;

 public:
  static constexpr unsigned rate = params::shake<Bits>::rate;

  void operator()(std::uint8_t* out, std::size_t out_pitch, std::size_t outlen,
                  const std::uint8_t* in, std::size_t in_pitch,
                  std::size_t inlen, cudaStream_t stream) const {
    global::shake<Bits><<<dg, db, ns, stream>>>(out, out_pitch, outlen, in,
                                                in_pitch, inlen, nin);
  }

  explicit shake(unsigned ninputs, unsigned warp_per_block)
      : dg(make_uint3((ninputs - 1 + warp_per_block) / warp_per_block, 1, 1)),
        db(make_uint3(32, warp_per_block, 1)),
        ns(rate * warp_per_block),
        nin(ninputs) {}

  shake() = delete;

  void* get_func() const {
    return reinterpret_cast<void*>(global::shake<Bits>);
  }
  dim3 get_grid_dim() const { return dg; }
  dim3 get_block_dim() const { return db; }
  unsigned get_shared_bytes() const { return ns; }
  using args_type =
      cuda_resource::kernel_args<std::uint8_t*, std::size_t, std::size_t,
                                 const std::uint8_t*, std::size_t, std::size_t,
                                 unsigned>;
  std::unique_ptr<args_type> generate_args(std::uint8_t* out,
                                           std::size_t out_pitch,
                                           std::size_t outlen,
                                           const std::uint8_t* in,
                                           std::size_t in_pitch,
                                           std::size_t inlen) const noexcept {
    return std::make_unique<args_type>(out, out_pitch, outlen, in, in_pitch,
                                       inlen, nin);
  }
};

}  // namespace atpqc_cuda::fips202_ws::host

#endif
