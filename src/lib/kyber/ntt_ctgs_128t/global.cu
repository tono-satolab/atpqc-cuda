//
// global.cu
// Kernels of NTT (128 threads version).
//
// Copyright (c) 2021 Tatsuki Ono
//
// This software is released under the MIT License.
// https://opensource.org/licenses/mit-license.php
//

#include <cstdint>

#include "../params.cuh"
#include "../reduce.cuh"
#include "device.cuh"
#include "global.cuh"

namespace atpqc_cuda::kyber::ntt_ctgs_128t::global {

__global__ void fwdntt(short2* poly, unsigned npolys) {
  device::fwdntt ntt;
  extern __shared__ std::int16_t shared_ptr[];

  if (unsigned pos = blockIdx.x * blockDim.y + threadIdx.y; pos < npolys) {
    poly += (params::n / 2) * pos;
    std::int16_t* tmp_poly = shared_ptr + (params::n / 2) * threadIdx.y;

    short2 p = poly[threadIdx.x];
    tmp_poly[threadIdx.x * 2 + 0] = p.x;
    tmp_poly[threadIdx.x * 2 + 1] = p.y;

    __syncthreads();

    ntt(tmp_poly);

    poly[threadIdx.x] =
        make_short2(reduce::barrett_reduce(tmp_poly[threadIdx.x * 2 + 0]),
                    reduce::barrett_reduce(tmp_poly[threadIdx.x * 2 + 1]));
  }
}

__global__ void invntt_tomont(short2* poly, unsigned npolys) {
  device::invntt intt;
  extern __shared__ std::int16_t shared_ptr[];

  if (unsigned pos = blockIdx.x * blockDim.y + threadIdx.y; pos < npolys) {
    poly += (params::n / 2) * pos;
    std::int16_t* tmp_poly = shared_ptr + (params::n / 2) * threadIdx.y;

    short2 p = poly[threadIdx.x];
    tmp_poly[threadIdx.x * 2 + 0] = p.x;
    tmp_poly[threadIdx.x * 2 + 1] = p.y;

    __syncthreads();

    intt(tmp_poly);

    constexpr std::int32_t f = reduce::mont_r *
                               (reduce::mont_r * (params::q - 1) *
                                ((params::q - 1) / 128) % params::q) %
                               params::q;
    poly[threadIdx.x] = make_short2(
        reduce::montgomery_reduce(f * tmp_poly[threadIdx.x * 2 + 0]),
        reduce::montgomery_reduce(f * tmp_poly[threadIdx.x * 2 + 1]));
  }
}

}  // namespace atpqc_cuda::kyber::ntt_ctgs_128t::global
