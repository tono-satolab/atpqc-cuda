//
// host.cuh
// Kernel wrapper for Hash functions.
//
// Copyright (c) 2021 Tatsuki Ono
//
// This software is released under the MIT License.
// https://opensource.org/licenses/mit-license.php
//

#ifndef ATPQC_CUDA_LIB_KYBER_SYMMETRIC_WS_HOST_CUH_
#define ATPQC_CUDA_LIB_KYBER_SYMMETRIC_WS_HOST_CUH_

#include <cstddef>
#include <cstdint>
#include <memory>

#include "../../cuda_resource.hpp"
#include "../../fips202_ws/global.cuh"
#include "../../fips202_ws/host.cuh"
#include "../../fips202_ws/params.cuh"
#include "../params.cuh"

namespace atpqc_cuda::kyber::symmetric_ws::host {

using hash_h = fips202_ws::host::sha3<256>;
using hash_g = fips202_ws::host::sha3<512>;

class kdf {
 private:
  dim3 dg;
  dim3 db;
  unsigned ns;
  unsigned nin;

 public:
  static constexpr unsigned rate = fips202_ws::params::shake<256>::rate;

  void operator()(std::uint8_t* out, std::size_t out_pitch,
                  const std::uint8_t* in, std::size_t in_pitch,
                  std::size_t inlen, cudaStream_t stream) const {
    fips202_ws::global::shake<256><<<dg, db, ns, stream>>>(
        out, out_pitch, params::ssbytes, in, in_pitch, inlen, nin);
  }

  explicit kdf(unsigned ninputs, unsigned warp_per_block)
      : dg(make_uint3((ninputs - 1 + warp_per_block) / warp_per_block, 1, 1)),
        db(make_uint3(32, warp_per_block, 1)),
        ns(rate * warp_per_block),
        nin(ninputs) {}

  kdf() = delete;

  void* get_func() const {
    return reinterpret_cast<void*>(fips202_ws::global::shake<256>);
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
                                           const std::uint8_t* in,
                                           std::size_t in_pitch,
                                           std::size_t inlen) const noexcept {
    return std::make_unique<args_type>(out, out_pitch, params::ssbytes, in,
                                       in_pitch, inlen, nin);
  }
};

}  // namespace atpqc_cuda::kyber::symmetric_ws::host

#endif
