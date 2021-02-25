//
// host.cuh
// Launches kernels of generation polynomials.
//
// Copyright (c) 2021 Tatsuki Ono
//
// This software is released under the MIT License.
// https://opensource.org/licenses/mit-license.php
//

#ifndef ATPQC_CUDA_LIB_KYBER_GENPOLY_WARP_HOST_CUH_
#define ATPQC_CUDA_LIB_KYBER_GENPOLY_WARP_HOST_CUH_

#include <cstddef>
#include <cstdint>
#include <memory>

#include "../../cuda_resource.hpp"
#include "global.cuh"
#include "kernel_params.cuh"

namespace atpqc_cuda::kyber::genpoly_warp::host {

template <unsigned K, bool Transposed>
class genmatrix {
 private:
  static constexpr unsigned k = K;
  static constexpr bool transposed = Transposed;
  using kp = kernel_params::genmatrix;

  dim3 dg;
  dim3 db;
  unsigned ns;
  unsigned npolys;

 public:
  void operator()(short2* polymat, const std::uint8_t* seed,
                  std::size_t seed_pitch, cudaStream_t stream) const {
    global::genmatrix<k, transposed>
        <<<dg, db, ns, stream>>>(polymat, seed, seed_pitch, npolys);
  }

  explicit genmatrix(unsigned ninputs, unsigned warp_per_block)
      : dg(make_uint3((ninputs * k * k - 1 + warp_per_block) / warp_per_block,
                      1, 1)),
        db(make_uint3(32, warp_per_block, 1)),
        ns(kp::smem_byte_per_warp * warp_per_block),
        npolys(k * k * ninputs) {}

  genmatrix() = delete;

  void* get_func() const {
    return reinterpret_cast<void*>(global::genmatrix<k, transposed>);
  }
  dim3 get_grid_dim() const { return dg; }
  dim3 get_block_dim() const { return db; }
  unsigned get_shared_bytes() const { return ns; }
  using args_type = cuda_resource::kernel_args<short2*, const std::uint8_t*,
                                               std::size_t, unsigned>;
  std::unique_ptr<args_type> generate_args(
      short2* polymat, const std::uint8_t* seed,
      std::size_t seed_pitch) const noexcept {
    return std::make_unique<args_type>(polymat, seed, seed_pitch, npolys);
  }
};

template <unsigned K>
using gena = genmatrix<K, false>;
template <unsigned K>
using genat = genmatrix<K, true>;

template <unsigned K, unsigned Eta>
class gennoise {
 private:
  static constexpr unsigned k = K;
  static constexpr unsigned eta = Eta;
  using kp = kernel_params::gennoise<eta>;

  dim3 dg;
  dim3 db;
  unsigned ns;
  unsigned npolys;

 public:
  void operator()(short2* poly, const std::uint8_t* seed,
                  std::size_t seed_pitch, std::uint8_t nonce_begin,
                  cudaStream_t stream) const {
    global::gennoise<k, eta>
        <<<dg, db, ns, stream>>>(poly, seed, seed_pitch, npolys, nonce_begin);
  }

  explicit gennoise(unsigned ninputs, unsigned warp_per_block)
      : dg(make_uint3((ninputs * k - 1 + warp_per_block) / warp_per_block, 1,
                      1)),
        db(make_uint3(32, warp_per_block, 1)),
        ns(kp::smem_byte_per_warp * warp_per_block),
        npolys(k * ninputs) {}

  gennoise() = delete;

  void* get_func() const {
    return reinterpret_cast<void*>(global::gennoise<k, eta>);
  }
  dim3 get_grid_dim() const { return dg; }
  dim3 get_block_dim() const { return db; }
  unsigned get_shared_bytes() const { return ns; }
  using args_type =
      cuda_resource::kernel_args<short2*, const std::uint8_t*, std::size_t,
                                 unsigned, std::uint8_t>;
  std::unique_ptr<args_type> generate_args(
      short2* poly, const std::uint8_t* seed, std::size_t seed_pitch,
      std::uint8_t nonce_begin) const noexcept {
    return std::make_unique<args_type>(poly, seed, seed_pitch, npolys,
                                       nonce_begin);
  }
};

}  // namespace atpqc_cuda::kyber::genpoly_warp::host

#endif
