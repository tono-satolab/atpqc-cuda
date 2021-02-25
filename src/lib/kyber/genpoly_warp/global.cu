//
// global.cu
// Kernels of generation polynomials.
//
// Copyright (c) 2021 Tatsuki Ono
//
// This software is released under the MIT License.
// https://opensource.org/licenses/mit-license.php
//

#include "device.cuh"
#include "global.cuh"
#include "kernel_params.cuh"

namespace atpqc_cuda::kyber::genpoly_warp::global {

template <unsigned K, bool Transposed>
__global__ void genmatrix(short2* poly, const std::uint8_t* seed,
                          std::size_t seed_pitch, unsigned npolys) {
  constexpr bool transposed = Transposed;
  constexpr unsigned k = K;
  using kp = kernel_params::genmatrix;

  extern __shared__ std::uint8_t shared_buf[];
  const unsigned polyid = blockIdx.x * blockDim.y + threadIdx.y;

  if (polyid < npolys) {
    std::int16_t* poly_buf = reinterpret_cast<std::int16_t*>(
        shared_buf + kp::smem_byte_per_warp * threadIdx.y);
    std::uint8_t* bytes_buf =
        reinterpret_cast<std::uint8_t*>(poly_buf) + kp::poly_bytes;

    poly += (params::n / 2) * polyid;
    seed += seed_pitch * (polyid / (k * k));
    unsigned x = polyid / k % k;
    unsigned y = polyid % k;

    symmetric_ws::device::state_type state;
    symmetric_ws::device::keccak_type keccak;
    symmetric_ws::device::xof xof;
    device::rej rej;

    bytes_buf[threadIdx.x] = seed[threadIdx.x];

    if (threadIdx.x == 0) {
      if constexpr (transposed) {
        bytes_buf[params::symbytes] = x;
        bytes_buf[params::symbytes + 1] = y;
      } else {
        bytes_buf[params::symbytes] = y;
        bytes_buf[params::symbytes + 1] = x;
      }
    }

    __syncwarp();

    state = xof.absorb(bytes_buf, keccak);
    state = xof.squeezeblocks(bytes_buf, kp::xof_nblocks, state, keccak);

    __syncwarp();

    unsigned ctr = rej(poly_buf, params::n, bytes_buf, kp::rej_bytes);
    unsigned buflen = kp::rej_bytes;
    while (ctr < params::n) {
      unsigned off = buflen % 3;
      if (threadIdx.x < off)
        bytes_buf[threadIdx.x] = bytes_buf[buflen - off + threadIdx.x];

      __syncwarp();

      state = xof.squeezeblocks(bytes_buf + off, 1, state, keccak);
      buflen = off + kp::xof_blockbytes;

      __syncwarp();

      ctr += rej(poly_buf + ctr, params::n - ctr, bytes_buf, buflen);
    }

    __syncwarp();

    poly[threadIdx.x + 0] = make_short2(poly_buf[2 * threadIdx.x + 0],
                                        poly_buf[2 * threadIdx.x + 1]);
    poly[threadIdx.x + 32] = make_short2(poly_buf[2 * threadIdx.x + 64],
                                         poly_buf[2 * threadIdx.x + 65]);
    poly[threadIdx.x + 64] = make_short2(poly_buf[2 * threadIdx.x + 128],
                                         poly_buf[2 * threadIdx.x + 129]);
    poly[threadIdx.x + 96] = make_short2(poly_buf[2 * threadIdx.x + 192],
                                         poly_buf[2 * threadIdx.x + 193]);
  }
}

template <unsigned K, unsigned Eta>
__global__ void gennoise(short2* poly, const std::uint8_t* seed,
                         std::size_t seed_pitch, unsigned npolys,
                         std::uint8_t nonce_begin) {
  constexpr unsigned k = K;
  constexpr unsigned eta = Eta;
  using kp = kernel_params::gennoise<eta>;

  extern __shared__ std::uint8_t shared_buf[];
  const unsigned polyid = blockIdx.x * blockDim.y + threadIdx.y;

  std::uint8_t* bytes_buf = shared_buf + kp::smem_byte_per_warp * threadIdx.y;

  if (polyid < npolys) {
    poly += (params::n / 2) * polyid;
    seed += seed_pitch * (polyid / k);

    symmetric_ws::device::keccak_type keccak;
    symmetric_ws::device::prf prf;
    device::cbd<eta> cbd;

    bytes_buf[threadIdx.x] = seed[threadIdx.x];
    if (threadIdx.x == 0)
      bytes_buf[params::symbytes] = nonce_begin + polyid % k;

    __syncwarp();

    prf(bytes_buf, kp::prf_nblocks, bytes_buf, kp::extseed_bytes, keccak);

    __syncwarp();

    cbd(poly, bytes_buf);
  }
}

template __global__ void genmatrix<2, true>(short2*, const std::uint8_t*,
                                            std::size_t, unsigned);
template __global__ void genmatrix<2, false>(short2*, const std::uint8_t*,
                                             std::size_t, unsigned);
template __global__ void genmatrix<3, true>(short2*, const std::uint8_t*,
                                            std::size_t, unsigned);
template __global__ void genmatrix<3, false>(short2*, const std::uint8_t*,
                                             std::size_t, unsigned);
template __global__ void genmatrix<4, true>(short2*, const std::uint8_t*,
                                            std::size_t, unsigned);
template __global__ void genmatrix<4, false>(short2*, const std::uint8_t*,
                                             std::size_t, unsigned);
template __global__ void gennoise<1, 2>(short2*, const std::uint8_t*,
                                        std::size_t, unsigned, std::uint8_t);
template __global__ void gennoise<2, 2>(short2*, const std::uint8_t*,
                                        std::size_t, unsigned, std::uint8_t);
template __global__ void gennoise<3, 2>(short2*, const std::uint8_t*,
                                        std::size_t, unsigned, std::uint8_t);
template __global__ void gennoise<4, 2>(short2*, const std::uint8_t*,
                                        std::size_t, unsigned, std::uint8_t);
template __global__ void gennoise<1, 3>(short2*, const std::uint8_t*,
                                        std::size_t, unsigned, std::uint8_t);
template __global__ void gennoise<2, 3>(short2*, const std::uint8_t*,
                                        std::size_t, unsigned, std::uint8_t);
template __global__ void gennoise<3, 3>(short2*, const std::uint8_t*,
                                        std::size_t, unsigned, std::uint8_t);
template __global__ void gennoise<4, 3>(short2*, const std::uint8_t*,
                                        std::size_t, unsigned, std::uint8_t);

}  // namespace atpqc_cuda::kyber::genpoly_warp::global
