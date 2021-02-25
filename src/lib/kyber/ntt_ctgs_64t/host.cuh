//
// host.cuh
// Launches kernels for NTT.
//
// Copyright (c) 2021 Tatsuki Ono
//
// This software is released under the MIT License.
// https://opensource.org/licenses/mit-license.php
//

#ifndef ATPQC_CUDA_LIB_KYBER_NTT_CTGS_64T_HOST_CUH_
#define ATPQC_CUDA_LIB_KYBER_NTT_CTGS_64T_HOST_CUH_

#include <memory>

#include "../../cuda_resource.hpp"
#include "../params.cuh"
#include "global.cuh"

namespace atpqc_cuda::kyber::ntt_ctgs_64t::host {

template <unsigned K>
class fwdntt {
 private:
  static constexpr unsigned k = K;

  dim3 dg;
  dim3 db;
  unsigned ns;
  unsigned np;

 public:
  void operator()(short2* poly, cudaStream_t stream) const {
    global::fwdntt<<<dg, db, ns, stream>>>(poly, np);
  }

  explicit fwdntt(unsigned ninputs, unsigned poly_per_block = 2)
      : dg(make_uint3((k * ninputs + poly_per_block - 1) / poly_per_block, 1,
                      1)),
        db(make_uint3(64, poly_per_block, 1)),
        ns(sizeof(short2) * params::n / 2 * poly_per_block),
        np(k * ninputs) {}

  fwdntt() = delete;

  void* get_func() const { return reinterpret_cast<void*>(global::fwdntt); }
  dim3 get_grid_dim() const { return dg; }
  dim3 get_block_dim() const { return db; }
  unsigned get_shared_bytes() const { return ns; }
  using args_type = cuda_resource::kernel_args<short2*, unsigned>;
  std::unique_ptr<args_type> generate_args(short2* poly) const noexcept {
    return std::make_unique<args_type>(poly, np);
  }
};

template <unsigned K>
class invntt_tomont {
 private:
  static constexpr unsigned k = K;

  dim3 dg;
  dim3 db;
  unsigned ns;
  unsigned np;

 public:
  void operator()(short2* poly, cudaStream_t stream) const {
    global::invntt_tomont<<<dg, db, ns, stream>>>(poly, np);
  }

  explicit invntt_tomont(unsigned ninputs, unsigned poly_per_block = 2)
      : dg(make_uint3((k * ninputs + poly_per_block - 1) / poly_per_block, 1,
                      1)),
        db(make_uint3(64, poly_per_block, 1)),
        ns(sizeof(short2) * params::n / 2 * poly_per_block),
        np(k * ninputs) {}

  invntt_tomont() = delete;

  void* get_func() const {
    return reinterpret_cast<void*>(global::invntt_tomont);
  }
  dim3 get_grid_dim() const { return dg; }
  dim3 get_block_dim() const { return db; }
  unsigned get_shared_bytes() const { return ns; }
  using args_type = cuda_resource::kernel_args<short2*, unsigned>;
  std::unique_ptr<args_type> generate_args(short2* poly) const noexcept {
    return std::make_unique<args_type>(poly, np);
  }
};

}  // namespace atpqc_cuda::kyber::ntt_ctgs_64t::host

#endif
