//
// global.cu
// Kernels for NTT.
//
// Copyright (c) 2021 Tatsuki Ono
//
// This software is released under the MIT License.
// https://opensource.org/licenses/mit-license.php
//

#include "../params.cuh"
#include "../reduce.cuh"
#include "device.cuh"
#include "global.cuh"

namespace atpqc_cuda::kyber::ntt_ctgs_64t::global {

__global__ void fwdntt(short2* poly, unsigned npolys) {
  device::fwdntt ntt;
  extern __shared__ short2 shared_ptr[];

  if (unsigned pos = blockIdx.x * blockDim.y + threadIdx.y; pos < npolys) {
    poly += (params::n / 2) * pos;
    short2* tmp_poly = shared_ptr + (params::n / 2) * threadIdx.y;

    tmp_poly[threadIdx.x + 0] = poly[threadIdx.x + 0];
    tmp_poly[threadIdx.x + 64] = poly[threadIdx.x + 64];

    __syncthreads();

    ntt(tmp_poly);

    short2 tmp0 = tmp_poly[threadIdx.x + 0];
    short2 tmp64 = tmp_poly[threadIdx.x + 64];
    tmp0 = make_short2(reduce::barrett_reduce(tmp0.x),
                       reduce::barrett_reduce(tmp0.y));
    tmp64 = make_short2(reduce::barrett_reduce(tmp64.x),
                        reduce::barrett_reduce(tmp64.y));
    poly[threadIdx.x + 0] = tmp0;
    poly[threadIdx.x + 64] = tmp64;
  }
}

__global__ void invntt_tomont(short2* poly, unsigned npolys) {
  device::invntt intt;
  extern __shared__ short2 shared_ptr[];

  if (unsigned pos = blockIdx.x * blockDim.y + threadIdx.y; pos < npolys) {
    poly += (params::n / 2) * pos;
    short2* tmp_poly = shared_ptr + (params::n / 2) * threadIdx.y;

    tmp_poly[threadIdx.x + 0] = poly[threadIdx.x + 0];
    tmp_poly[threadIdx.x + 64] = poly[threadIdx.x + 64];

    __syncthreads();

    intt(tmp_poly);

    constexpr std::int32_t f = reduce::mont_r *
                               (reduce::mont_r * (params::q - 1) *
                                ((params::q - 1) / 128) % params::q) %
                               params::q;
    short2 tmp0 = tmp_poly[threadIdx.x + 0];
    short2 tmp64 = tmp_poly[threadIdx.x + 64];
    tmp0 = make_short2(reduce::montgomery_reduce(f * tmp0.x),
                       reduce::montgomery_reduce(f * tmp0.y));
    tmp64 = make_short2(reduce::montgomery_reduce(f * tmp64.x),
                        reduce::montgomery_reduce(f * tmp64.y));
    poly[threadIdx.x + 0] = tmp0;
    poly[threadIdx.x + 64] = tmp64;
  }
}

}  // namespace atpqc_cuda::kyber::ntt_ctgs_64t::global
