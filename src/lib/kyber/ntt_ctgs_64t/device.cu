//
// device.cu
// Device functions for NTT.
//
// Copyright (c) 2021 Tatsuki Ono
//
// This software is released under the MIT License.
// https://opensource.org/licenses/mit-license.php
//

#include "../reduce.cuh"
#include "../zetas_table.cuh"
#include "device.cuh"

namespace atpqc_cuda::kyber::ntt_ctgs_64t::device {

namespace {
constexpr unsigned ntt_n = 2;
constexpr unsigned log_n = 8;
constexpr unsigned log_ntt_n = 1;
constexpr unsigned log_diff = log_n - log_ntt_n;
}  // namespace

__device__ void fwdntt::operator()(short2* shared_poly) const {
  unsigned tid = threadIdx.x;
  unsigned k = 1;
  unsigned lg = log_diff - 1;

  for (std::size_t len = params::n / 2 / 2; len >= ntt_n / 2; len >>= 1) {
    unsigned group = tid >> lg;
    unsigned addr = tid + (group << lg);

    short2 a = shared_poly[addr];
    short2 b = shared_poly[addr + len];
    int zeta = zetas_table::zetas[k + group];

    short tmpx = reduce::montgomery_reduce(zeta * b.x);
    short tmpy = reduce::montgomery_reduce(zeta * b.y);

    b = make_short2(a.x - tmpx, a.y - tmpy);
    a = make_short2(a.x + tmpx, a.y + tmpy);

    shared_poly[addr] = a;
    shared_poly[addr + len] = b;

    k <<= 1;
    --lg;

    __syncthreads();
  }
}

__device__ void invntt::operator()(short2* shared_poly) const {
  unsigned tid = threadIdx.x;
  unsigned k = params::n / ntt_n - 1;
  unsigned lg = 0;

  for (std::size_t len = ntt_n / 2; len < params::n / 2; len <<= 1) {
    unsigned group = tid >> lg;
    unsigned addr = tid + (group << lg);

    short2 a = shared_poly[addr];
    short2 b = shared_poly[addr + len];
    int zeta = zetas_table::zetas[k - group];

    short tmpx = a.x;
    short tmpy = a.y;
    a = make_short2(reduce::barrett_reduce(tmpx + b.x),
                    reduce::barrett_reduce(tmpy + b.y));
    tmpx = b.x - tmpx;
    tmpy = b.y - tmpy;
    b = make_short2(reduce::montgomery_reduce(zeta * tmpx),
                    reduce::montgomery_reduce(zeta * tmpy));

    shared_poly[addr] = a;
    shared_poly[addr + len] = b;

    k >>= 1;
    ++lg;

    __syncthreads();
  }
}

}  // namespace atpqc_cuda::kyber::ntt_ctgs_64t::device
