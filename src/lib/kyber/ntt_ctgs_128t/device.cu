//
// device.cu
// Device functions for NTT (128 threads version).
//
// Copyright (c) 2021 Tatsuki Ono
//
// This software is released under the MIT License.
// https://opensource.org/licenses/mit-license.php
//

#include "../reduce.cuh"
#include "../zetas_table.cuh"
#include "device.cuh"

namespace atpqc_cuda::kyber::ntt_ctgs_128t::device {

namespace {
constexpr unsigned log_n = 8;
constexpr unsigned frequency_n = 2;
constexpr unsigned log_frequency_n = 1;
constexpr unsigned log_diff = log_n - log_frequency_n;
}  // namespace

__device__ void fwdntt::operator()(std::int16_t* shared_poly) const {
  unsigned tid = threadIdx.x;
  unsigned k = 1;
  unsigned lg = log_diff;

  for (std::size_t len = params::n / 2; len >= frequency_n; len >>= 1) {
    unsigned group = tid >> lg;
    unsigned addr = tid + (group << lg);

    std::int16_t a = shared_poly[addr];
    std::int16_t b = shared_poly[addr + len];
    std::int32_t zeta = zetas_table::zetas[k + group];

    std::int16_t tmp = reduce::montgomery_reduce(zeta * b);
    b = a - tmp;
    a = a + tmp;

    shared_poly[addr] = a;
    shared_poly[addr + len] = b;

    k <<= 1;
    --lg;

    __syncthreads();
  }
}

__device__ void invntt::operator()(std::int16_t* shared_poly) const {
  unsigned tid = threadIdx.x;
  unsigned k = params::n / frequency_n - 1;
  unsigned lg = 1;

  for (std::size_t len = frequency_n; len < params::n; len <<= 1) {
    unsigned group = tid >> lg;
    unsigned addr = tid + (group << lg);

    std::int16_t a = shared_poly[addr];
    std::int16_t b = shared_poly[addr + len];
    std::int32_t zeta = zetas_table::zetas[k - group];

    std::int16_t tmp = a;
    a = reduce::barrett_reduce(tmp + b);
    tmp = b - tmp;
    b = reduce::montgomery_reduce(zeta * tmp);

    shared_poly[addr] = a;
    shared_poly[addr + len] = b;

    k >>= 1;
    ++lg;

    __syncthreads();
  }
}

}  // namespace atpqc_cuda::kyber::ntt_ctgs_128t::device
