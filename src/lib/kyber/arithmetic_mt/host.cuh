//
// host.cuh
// Launches module arithetic kernels.
//
// Copyright (c) 2021 Tatsuki Ono
//
// This software is released under the MIT License.
// https://opensource.org/licenses/mit-license.php
//

#ifndef ATPQC_CUDA_LIB_KYBER_ARITHMETIC_MT_HOST_CUH_
#define ATPQC_CUDA_LIB_KYBER_ARITHMETIC_MT_HOST_CUH_

#include <memory>

#include "../../cuda_resource.hpp"
#include "global.cuh"
#include "kernel_params.cuh"

namespace atpqc_cuda::kyber::arithmetic_mt::host {

template <unsigned K>
class mattimesvec_tomont_plusvec {
 private:
  static constexpr unsigned k = K;
  using kp = kernel_params::mattimes<k>;

  dim3 dg;
  dim3 db;
  unsigned ns;

 public:
  void operator()(short2* vec_out, const short2* mat, const short2* vec1,
                  const short2* vec2, cudaStream_t stream) const {
    global::mattimesvec_tomont_plusvec<k>
        <<<dg, db, ns, stream>>>(vec_out, mat, vec1, vec2);
  }

  explicit mattimesvec_tomont_plusvec(unsigned ninputs,
                                      unsigned coeff_per_block = 64)
      : dg(make_uint3(params::n / 2 / coeff_per_block, ninputs, 1)),
        db(make_uint3(coeff_per_block, k, 1)),
        ns(kp::smem_byte_per_coeff * coeff_per_block) {}

  mattimesvec_tomont_plusvec() = delete;

  void* get_func() const {
    return reinterpret_cast<void*>(global::mattimesvec_tomont_plusvec<k>);
  }
  dim3 get_grid_dim() const { return dg; }
  dim3 get_block_dim() const { return db; }
  unsigned get_shared_bytes() const { return ns; }
  using args_type = cuda_resource::kernel_args<short2*, const short2*,
                                               const short2*, const short2*>;
  std::unique_ptr<args_type> generate_args(short2* vec_out, const short2* mat,
                                           const short2* vec1,
                                           const short2* vec2) const noexcept {
    return std::make_unique<args_type>(vec_out, mat, vec1, vec2);
  }
};

template <unsigned K>
class mattimesvec {
 private:
  static constexpr unsigned k = K;
  using kp = kernel_params::mattimes<k>;

  dim3 dg;
  dim3 db;
  unsigned ns;

 public:
  void operator()(short2* vec_out, const short2* mat, const short2* vec,
                  cudaStream_t stream) const {
    global::mattimesvec<k><<<dg, db, ns, stream>>>(vec_out, mat, vec);
  }

  explicit mattimesvec(unsigned ninputs, unsigned coeff_per_block = 64)
      : dg(make_uint3(params::n / 2 / coeff_per_block, ninputs, 1)),
        db(make_uint3(coeff_per_block, k, 1)),
        ns(kp::smem_byte_per_coeff * coeff_per_block) {}

  mattimesvec() = delete;

  void* get_func() const {
    return reinterpret_cast<void*>(global::mattimesvec<k>);
  }
  dim3 get_grid_dim() const { return dg; }
  dim3 get_block_dim() const { return db; }
  unsigned get_shared_bytes() const { return ns; }
  using args_type =
      cuda_resource::kernel_args<short2*, const short2*, const short2*>;
  std::unique_ptr<args_type> generate_args(short2* vec_out, const short2* mat,
                                           const short2* vec) const noexcept {
    return std::make_unique<args_type>(vec_out, mat, vec);
  }
};

template <unsigned K>
class vectimesvec {
 private:
  static constexpr unsigned k = K;

  dim3 dg;
  dim3 db;
  unsigned ns;

 public:
  void operator()(short2* poly_out, const short2* vec1, const short2* vec2,
                  cudaStream_t stream) const {
    global::vectimesvec<k><<<dg, db, ns, stream>>>(poly_out, vec1, vec2);
  }

  explicit vectimesvec(unsigned ninputs)
      : dg(make_uint3(ninputs, 1, 1)),
        db(make_uint3(params::n / 2, 1, 1)),
        ns(0) {}

  vectimesvec() = delete;

  void* get_func() const {
    return reinterpret_cast<void*>(global::vectimesvec<k>);
  }
  dim3 get_grid_dim() const { return dg; }
  dim3 get_block_dim() const { return db; }
  unsigned get_shared_bytes() const { return ns; }
  using args_type =
      cuda_resource::kernel_args<short2*, const short2*, const short2*>;
  std::unique_ptr<args_type> generate_args(short2* poly_out, const short2* vec1,
                                           const short2* vec2) const noexcept {
    return std::make_unique<args_type>(poly_out, vec1, vec2);
  }
};

template <unsigned K>
class vecadd2 {
 private:
  static constexpr unsigned k = K;

  dim3 dg;
  dim3 db;
  unsigned ns;

 public:
  void operator()(short2* vec_out, const short2* vec1, const short2* vec2,
                  cudaStream_t stream) const {
    global::add2<<<dg, db, ns, stream>>>(vec_out, vec1, vec2);
  }

  explicit vecadd2(unsigned ninputs)
      : dg(make_uint3(k * ninputs, 1, 1)),
        db(make_uint3(params::n / 2, 1, 1)),
        ns(0) {}

  vecadd2() = delete;

  void* get_func() const { return reinterpret_cast<void*>(global::add2); }
  dim3 get_grid_dim() const { return dg; }
  dim3 get_block_dim() const { return db; }
  unsigned get_shared_bytes() const { return ns; }
  using args_type =
      cuda_resource::kernel_args<short2*, const short2*, const short2*>;
  std::unique_ptr<args_type> generate_args(short2* vec_out, const short2* vec1,
                                           const short2* vec2) const noexcept {
    return std::make_unique<args_type>(vec_out, vec1, vec2);
  }
};

class polyadd3 {
 private:
  dim3 dg;
  dim3 db;
  unsigned ns;

 public:
  void operator()(short2* poly_out, const short2* poly1, const short2* poly2,
                  const short2* poly3, cudaStream_t stream) const {
    global::add3<<<dg, db, ns, stream>>>(poly_out, poly1, poly2, poly3);
  }

  explicit polyadd3(unsigned ninputs)
      : dg(make_uint3(ninputs, 1, 1)),
        db(make_uint3(params::n / 2, 1, 1)),
        ns(0) {}

  polyadd3() = delete;

  void* get_func() const { return reinterpret_cast<void*>(global::add3); }
  dim3 get_grid_dim() const { return dg; }
  dim3 get_block_dim() const { return db; }
  unsigned get_shared_bytes() const { return ns; }
  using args_type = cuda_resource::kernel_args<short2*, const short2*,
                                               const short2*, const short2*>;
  std::unique_ptr<args_type> generate_args(short2* poly_out,
                                           const short2* poly1,
                                           const short2* poly2,
                                           const short2* poly3) const noexcept {
    return std::make_unique<args_type>(poly_out, poly1, poly2, poly3);
  }
};

class polysub {
 private:
  dim3 dg;
  dim3 db;
  unsigned ns;

 public:
  void operator()(short2* poly_out, const short2* poly1, const short2* poly2,
                  cudaStream_t stream) const {
    global::sub<<<dg, db, ns, stream>>>(poly_out, poly1, poly2);
  }

  explicit polysub(unsigned ninputs)
      : dg(make_uint3(ninputs, 1, 1)),
        db(make_uint3(params::n / 2, 1, 1)),
        ns(0) {}

  polysub() = delete;

  void* get_func() const { return reinterpret_cast<void*>(global::sub); }
  dim3 get_grid_dim() const { return dg; }
  dim3 get_block_dim() const { return db; }
  unsigned get_shared_bytes() const { return ns; }
  using args_type =
      cuda_resource::kernel_args<short2*, const short2*, const short2*>;
  std::unique_ptr<args_type> generate_args(short2* poly_out,
                                           const short2* poly1,
                                           const short2* poly2) const noexcept {
    return std::make_unique<args_type>(poly_out, poly1, poly2);
  }
};

}  // namespace atpqc_cuda::kyber::arithmetic_mt::host

#endif
