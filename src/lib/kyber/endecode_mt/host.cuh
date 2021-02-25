//
// host.cuh
// Launches kernels for encoding/decoding polynomials.
//
// Copyright (c) 2021 Tatsuki Ono
//
// This software is released under the MIT License.
// https://opensource.org/licenses/mit-license.php
//

#ifndef ATPQC_CUDA_LIB_KYBER_ENDECODE_MT_HOST_CUH_
#define ATPQC_CUDA_LIB_KYBER_ENDECODE_MT_HOST_CUH_

#include <cstddef>
#include <cstdint>
#include <memory>

#include "../../cuda_resource.hpp"
#include "global.cuh"
#include "kernel_params.cuh"

namespace atpqc_cuda::kyber::endecode_mt::host {

template <unsigned Dv>
class poly_compress {
 private:
  static constexpr unsigned dv = Dv;
  using kp = kernel_params::poly_de_compress<dv>;

  dim3 dg;
  dim3 db;
  unsigned ns;
  unsigned nin;

 public:
  void operator()(std::uint8_t* cbytes, std::size_t cbytes_pitch,
                  const short2* poly, cudaStream_t stream) const {
    global::poly_compress<dv>
        <<<dg, db, ns, stream>>>(cbytes, cbytes_pitch, poly, nin);
  }

  explicit poly_compress(unsigned ninputs,
                         unsigned input_per_block = 128 / kp::thread_per_input)
      : dg(make_uint3((ninputs + input_per_block - 1) / input_per_block, 1, 1)),
        db(make_uint3(kp::thread_per_input, input_per_block, 1)),
        ns(0),
        nin(ninputs) {}

  poly_compress() = delete;

  void* get_func() const {
    return reinterpret_cast<void*>(global::poly_compress<dv>);
  }
  dim3 get_grid_dim() const { return dg; }
  dim3 get_block_dim() const { return db; }
  unsigned get_shared_bytes() const { return ns; }
  using args_type = cuda_resource::kernel_args<std::uint8_t*, std::size_t,
                                               const short2*, unsigned>;
  std::unique_ptr<args_type> generate_args(std::uint8_t* cbytes,
                                           std::size_t cbytes_pitch,
                                           const short2* poly) const noexcept {
    return std::make_unique<args_type>(cbytes, cbytes_pitch, poly, nin);
  }
};

template <unsigned Dv>
class poly_decompress {
 private:
  static constexpr unsigned dv = Dv;
  using kp = kernel_params::poly_de_compress<dv>;

  dim3 dg;
  dim3 db;
  unsigned ns;
  unsigned nin;

 public:
  void operator()(short2* poly, const std::uint8_t* cbytes,
                  std::size_t cbytes_pitch, cudaStream_t stream) const {
    global::poly_decompress<dv>
        <<<dg, db, ns, stream>>>(poly, cbytes, cbytes_pitch, nin);
  }

  explicit poly_decompress(unsigned ninputs,
                           unsigned input_per_block = 128 /
                                                      kp::thread_per_input)
      : dg(make_uint3((ninputs + input_per_block - 1) / input_per_block, 1, 1)),
        db(make_uint3(kp::thread_per_input, input_per_block, 1)),
        ns(0),
        nin(ninputs) {}

  poly_decompress() = delete;

  void* get_func() const {
    return reinterpret_cast<void*>(global::poly_decompress<dv>);
  }
  dim3 get_grid_dim() const { return dg; }
  dim3 get_block_dim() const { return db; }
  unsigned get_shared_bytes() const { return ns; }
  using args_type = cuda_resource::kernel_args<short2*, const std::uint8_t*,
                                               std::size_t, unsigned>;
  std::unique_ptr<args_type> generate_args(
      short2* poly, const std::uint8_t* cbytes,
      std::size_t cbytes_pitch) const noexcept {
    return std::make_unique<args_type>(poly, cbytes, cbytes_pitch, nin);
  }
};

template <unsigned K, unsigned Du>
class polyvec_compress {
 private:
  static constexpr unsigned k = K;
  static constexpr unsigned du = Du;
  using kp = kernel_params::polyvec_de_compress<k, du>;

  dim3 dg;
  dim3 db;
  unsigned ns;
  unsigned nin;

 public:
  void operator()(std::uint8_t* cbytes, std::size_t cbytes_pitch,
                  const short2* polyvec, cudaStream_t stream) const {
    global::polyvec_compress<k, du>
        <<<dg, db, ns, stream>>>(cbytes, cbytes_pitch, polyvec, nin);
  }

  explicit polyvec_compress(unsigned ninputs, unsigned input_per_block = 1)
      : dg(make_uint3((ninputs + input_per_block - 1) / input_per_block, 1, 1)),
        db(make_uint3(kp::thread_per_input, input_per_block, 1)),
        ns(0),
        nin(ninputs) {}

  polyvec_compress() = delete;

  void* get_func() const {
    return reinterpret_cast<void*>(global::polyvec_compress<k, du>);
  }
  dim3 get_grid_dim() const { return dg; }
  dim3 get_block_dim() const { return db; }
  unsigned get_shared_bytes() const { return ns; }
  using args_type = cuda_resource::kernel_args<std::uint8_t*, std::size_t,
                                               const short2*, unsigned>;
  std::unique_ptr<args_type> generate_args(
      std::uint8_t* cbytes, std::size_t cbytes_pitch,
      const short2* polyvec) const noexcept {
    return std::make_unique<args_type>(cbytes, cbytes_pitch, polyvec, nin);
  }
};

template <unsigned K, unsigned Du>
class polyvec_decompress {
 private:
  static constexpr unsigned k = K;
  static constexpr unsigned du = Du;
  using kp = kernel_params::polyvec_de_compress<k, du>;

  dim3 dg;
  dim3 db;
  unsigned ns;
  unsigned nin;

 public:
  void operator()(short2* polyvec, const std::uint8_t* cbytes,
                  std::size_t cbytes_pitch, cudaStream_t stream) const {
    global::polyvec_decompress<k, du>
        <<<dg, db, ns, stream>>>(polyvec, cbytes, cbytes_pitch, nin);
  }

  explicit polyvec_decompress(unsigned ninputs, unsigned input_per_block = 1)
      : dg(make_uint3((ninputs + input_per_block - 1) / input_per_block, 1, 1)),
        db(make_uint3(kp::thread_per_input, input_per_block, 1)),
        ns(0),
        nin(ninputs) {}

  polyvec_decompress() = delete;

  void* get_func() const {
    return reinterpret_cast<void*>(global::polyvec_decompress<k, du>);
  }
  dim3 get_grid_dim() const { return dg; }
  dim3 get_block_dim() const { return db; }
  unsigned get_shared_bytes() const { return ns; }
  using args_type = cuda_resource::kernel_args<short2*, const std::uint8_t*,
                                               std::size_t, unsigned>;
  std::unique_ptr<args_type> generate_args(
      short2* polyvec, const std::uint8_t* cbytes,
      std::size_t cbytes_pitch) const noexcept {
    return std::make_unique<args_type>(polyvec, cbytes, cbytes_pitch, nin);
  }
};

template <unsigned K>
class polyvec_tobytes {
 private:
  static constexpr unsigned k = K;
  using kp = kernel_params::polyvec_bytes<k>;

  dim3 dg;
  dim3 db;
  unsigned ns;
  unsigned nin;

 public:
  void operator()(std::uint8_t* bytes, std::size_t bytes_pitch,
                  const short2* polyvec, cudaStream_t stream) const {
    global::polyvec_tobytes<k>
        <<<dg, db, ns, stream>>>(bytes, bytes_pitch, polyvec, nin);
  }

  explicit polyvec_tobytes(unsigned ninputs, unsigned input_per_block = 1)
      : dg(make_uint3((ninputs + input_per_block - 1) / input_per_block, 1, 1)),
        db(make_uint3(kp::thread_per_input, input_per_block, 1)),
        ns(0),
        nin(ninputs) {}

  polyvec_tobytes() = delete;

  void* get_func() const {
    return reinterpret_cast<void*>(global::polyvec_tobytes<k>);
  }
  dim3 get_grid_dim() const { return dg; }
  dim3 get_block_dim() const { return db; }
  unsigned get_shared_bytes() const { return ns; }
  using args_type = cuda_resource::kernel_args<std::uint8_t*, std::size_t,
                                               const short2*, unsigned>;
  std::unique_ptr<args_type> generate_args(
      std::uint8_t* bytes, std::size_t bytes_pitch,
      const short2* polyvec) const noexcept {
    return std::make_unique<args_type>(bytes, bytes_pitch, polyvec, nin);
  }
};

template <unsigned K>
class polyvec_frombytes {
 private:
  static constexpr unsigned k = K;
  using kp = kernel_params::polyvec_bytes<k>;

  dim3 dg;
  dim3 db;
  unsigned ns;
  unsigned nin;

 public:
  void operator()(short2* polyvec, const std::uint8_t* bytes,
                  std::size_t bytes_pitch, cudaStream_t stream) const {
    global::polyvec_frombytes<k>
        <<<dg, db, ns, stream>>>(polyvec, bytes, bytes_pitch, nin);
  }

  explicit polyvec_frombytes(unsigned ninputs, unsigned input_per_block = 1)
      : dg(make_uint3((ninputs + input_per_block - 1) / input_per_block, 1, 1)),
        db(make_uint3(kp::thread_per_input, input_per_block, 1)),
        ns(0),
        nin(ninputs) {}

  polyvec_frombytes() = delete;

  void* get_func() const {
    return reinterpret_cast<void*>(global::polyvec_frombytes<k>);
  }
  dim3 get_grid_dim() const { return dg; }
  dim3 get_block_dim() const { return db; }
  unsigned get_shared_bytes() const { return ns; }
  using args_type = cuda_resource::kernel_args<short2*, const std::uint8_t*,
                                               std::size_t, unsigned>;
  std::unique_ptr<args_type> generate_args(
      short2* polyvec, const std::uint8_t* bytes,
      std::size_t bytes_pitch) const noexcept {
    return std::make_unique<args_type>(polyvec, bytes, bytes_pitch, nin);
  }
};

class poly_frommsg {
 private:
  using kp = kernel_params::poly_msg;

  dim3 dg;
  dim3 db;
  unsigned ns;
  unsigned nin;

 public:
  void operator()(short2* poly, const std::uint8_t* msg, std::size_t msg_pitch,
                  cudaStream_t stream) const {
    global::poly_frommsg<<<dg, db, ns, stream>>>(poly, msg, msg_pitch, nin);
  }

  explicit poly_frommsg(unsigned ninputs, unsigned input_per_block = 4)
      : dg(make_uint3((ninputs + input_per_block - 1) / input_per_block, 1, 1)),
        db(make_uint3(kp::thread_per_input, input_per_block, 1)),
        ns(0),
        nin(ninputs) {}

  poly_frommsg() = delete;

  void* get_func() const {
    return reinterpret_cast<void*>(global::poly_frommsg);
  }
  dim3 get_grid_dim() const { return dg; }
  dim3 get_block_dim() const { return db; }
  unsigned get_shared_bytes() const { return ns; }
  using args_type = cuda_resource::kernel_args<short2*, const std::uint8_t*,
                                               std::size_t, unsigned>;
  std::unique_ptr<args_type> generate_args(
      short2* poly, const std::uint8_t* msg,
      std::size_t msg_pitch) const noexcept {
    return std::make_unique<args_type>(poly, msg, msg_pitch, nin);
  }
};

class poly_tomsg {
 private:
  using kp = kernel_params::poly_msg;

  dim3 dg;
  dim3 db;
  unsigned ns;
  unsigned nin;

 public:
  void operator()(std::uint8_t* msg, std::size_t msg_pitch, const short2* poly,
                  cudaStream_t stream) const {
    global::poly_tomsg<<<dg, db, ns, stream>>>(msg, msg_pitch, poly, nin);
  }

  explicit poly_tomsg(unsigned ninputs, unsigned input_per_block = 4)
      : dg(make_uint3((ninputs + input_per_block - 1) / input_per_block, 1, 1)),
        db(make_uint3(kp::thread_per_input, input_per_block, 1)),
        ns(0),
        nin(ninputs) {}

  poly_tomsg() = delete;

  void* get_func() const { return reinterpret_cast<void*>(global::poly_tomsg); }
  dim3 get_grid_dim() const { return dg; }
  dim3 get_block_dim() const { return db; }
  unsigned get_shared_bytes() const { return ns; }
  using args_type = cuda_resource::kernel_args<std::uint8_t*, std::size_t,
                                               const short2*, unsigned>;
  std::unique_ptr<args_type> generate_args(std::uint8_t* msg,
                                           std::size_t msg_pitch,
                                           const short2* poly) const noexcept {
    return std::make_unique<args_type>(msg, msg_pitch, poly, nin);
  }
};

}  // namespace atpqc_cuda::kyber::endecode_mt::host

#endif
