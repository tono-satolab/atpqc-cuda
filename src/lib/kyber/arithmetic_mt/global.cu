//
// global.cu
// Kernel of addition, subtraction, and multiplication
// for matrices, modules, and polynomials.
//
// Copyright (c) 2021 Tatsuki Ono
//
// This software is released under the MIT License.
// https://opensource.org/licenses/mit-license.php
//

#include <cstdint>

#include "../params.cuh"
#include "../reduce.cuh"
#include "../zetas_table.cuh"
#include "global.cuh"
#include "kernel_params.cuh"

namespace atpqc_cuda::kyber::arithmetic_mt::global {

namespace {

__device__ short fqmul(std::int32_t a, std::int32_t b) noexcept {
  return reduce::montgomery_reduce(a * b);
}

__device__ void basemul(short& out0, short& out1, short lhs0, short lhs1,
                        short rhs0, short rhs1, short zeta) noexcept {
  short r0 = fqmul(lhs1, rhs1);
  r0 = fqmul(r0, zeta);
  r0 += fqmul(lhs0, rhs0);

  short r1 = fqmul(lhs0, rhs1);
  r1 += fqmul(lhs1, rhs0);

  out0 = r0;
  out1 = r1;
}

}  // namespace

template <unsigned K>
__global__ void mattimesvec_tomont_plusvec(short2* vec_out, const short2* mat,
                                           const short2* vec1,
                                           const short2* vec2) {
  constexpr unsigned k = K;

  extern __shared__ short2 tmp_shared[];

  const unsigned coeffid = blockIdx.x * blockDim.x + threadIdx.x;

  mat += (k * k * params::n / 2) * blockIdx.y + (params::n / 2) * threadIdx.y +
         coeffid;
  vec_out += (k * params::n / 2) * blockIdx.y + (params::n / 2) * threadIdx.y +
             coeffid;
  vec1 += (k * params::n / 2) * blockIdx.y + (params::n / 2) * threadIdx.y +
          coeffid;
  vec2 += (k * params::n / 2) * blockIdx.y + (params::n / 2) * threadIdx.y +
          coeffid;
  short2* tmp_mat_pointwise_ptr =
      tmp_shared + blockDim.x * threadIdx.y + threadIdx.x;
  short2* tmp_mat_acc_ptr =
      tmp_shared + (k * blockDim.x) * threadIdx.y + threadIdx.x;

  {
    short zeta = zetas_table::zetas[64 + (coeffid >> 1)];
    short b = coeffid & 0b1;
    zeta = (zeta ^ (-b)) + b;

    short2 v = *vec1;
    short rhs_x = v.x;
    short rhs_y = v.y;
    for (unsigned ki = 0; ki < k; ++ki) {
      short2 m = mat[(k * params::n / 2) * ki];
      short lhs_x = m.x;
      short lhs_y = m.y;
      short tmp_x, tmp_y;
      basemul(tmp_x, tmp_y, lhs_x, lhs_y, rhs_x, rhs_y, zeta);
      tmp_mat_pointwise_ptr[(k * blockDim.x) * ki] = make_short2(tmp_x, tmp_y);
    }
  }  // pointwise_multiplication

  __syncthreads();

  short sum0 = 0;
  short sum1 = 0;
  {
    for (unsigned ki = 0; ki < k; ++ki) {
      short2 mtmp = tmp_mat_acc_ptr[blockDim.x * ki];
      sum0 += mtmp.x;
      sum1 += mtmp.y;
    }
    constexpr std::int16_t mont_r_squared = (1ULL << 32) % params::q;
    sum0 = reduce::montgomery_reduce(static_cast<std::int32_t>(sum0) *
                                     mont_r_squared);
    sum1 = reduce::montgomery_reduce(static_cast<std::int32_t>(sum1) *
                                     mont_r_squared);
  }  // accumulation

  short2 vtmp = *vec2;
  sum0 = reduce::barrett_reduce(sum0 + vtmp.x);
  sum1 = reduce::barrett_reduce(sum1 + vtmp.y);
  *vec_out = make_short2(sum0, sum1);
}

template <unsigned K>
__global__ void mattimesvec(short2* vec_out, const short2* mat,
                            const short2* vec) {
  constexpr unsigned k = K;

  extern __shared__ short2 tmp_shared[];

  const unsigned coeffid = blockIdx.x * blockDim.x + threadIdx.x;

  mat += (k * k * params::n / 2) * blockIdx.y + (params::n / 2) * threadIdx.y +
         coeffid;
  vec_out += (k * params::n / 2) * blockIdx.y + (params::n / 2) * threadIdx.y +
             coeffid;
  vec += (k * params::n / 2) * blockIdx.y + (params::n / 2) * threadIdx.y +
         coeffid;
  short2* tmp_mat_pointwise_ptr =
      tmp_shared + blockDim.x * threadIdx.y + threadIdx.x;
  short2* tmp_mat_acc_ptr =
      tmp_shared + (k * blockDim.x) * threadIdx.y + threadIdx.x;

  {
    short zeta = zetas_table::zetas[64 + (coeffid >> 1)];
    short b = coeffid & 0b1;
    zeta = (zeta ^ (-b)) + b;

    short2 v = *vec;
    short rhs_x = v.x;
    short rhs_y = v.y;
    for (unsigned ki = 0; ki < k; ++ki) {
      short2 m = mat[(k * params::n / 2) * ki];
      short lhs_x = m.x;
      short lhs_y = m.y;
      short tmp_x, tmp_y;
      basemul(tmp_x, tmp_y, lhs_x, lhs_y, rhs_x, rhs_y, zeta);
      tmp_mat_pointwise_ptr[(k * blockDim.x) * ki] = make_short2(tmp_x, tmp_y);
    }
  }  // pointwise_multiplication

  __syncthreads();

  short sum0 = 0;
  short sum1 = 0;
  {
    for (unsigned ki = 0; ki < k; ++ki) {
      short2 mtmp = tmp_mat_acc_ptr[blockDim.x * ki];
      sum0 += mtmp.x;
      sum1 += mtmp.y;
    }
    sum0 = reduce::barrett_reduce(sum0);
    sum1 = reduce::barrett_reduce(sum1);
  }  // accumulation
  *vec_out = make_short2(sum0, sum1);
}

template <unsigned K>
__global__ void vectimesvec(short2* poly_out, const short2* vec1,
                            const short2* vec2) {
  constexpr unsigned k = K;

  poly_out += (params::n / 2) * blockIdx.x + threadIdx.x;
  vec1 += (k * params::n / 2) * blockIdx.x + threadIdx.x;
  vec2 += (k * params::n / 2) * blockIdx.x + threadIdx.x;

  short zeta = zetas_table::zetas[64 + (threadIdx.x >> 1)];
  std::int16_t b = threadIdx.x & 0b1;
  zeta = (zeta ^ (-b)) + b;

  short sum0 = 0;
  short sum1 = 0;
  for (unsigned ki = 0; ki < k; ++ki) {
    short2 v1 = vec1[(params::n / 2) * ki];
    short2 v2 = vec2[(params::n / 2) * ki];
    short lhs_x = v1.x;
    short lhs_y = v1.y;
    short rhs_x = v2.x;
    short rhs_y = v2.y;
    short tmp_x, tmp_y;
    basemul(tmp_x, tmp_y, lhs_x, lhs_y, rhs_x, rhs_y, zeta);
    sum0 += tmp_x;
    sum1 += tmp_y;
  }
  *poly_out = make_short2(sum0, sum1);
}

__global__ void add2(short2* poly_out, const short2* poly1,
                     const short2* poly2) {
  poly_out += (params::n / 2) * blockIdx.x + threadIdx.x;
  poly1 += (params::n / 2) * blockIdx.x + threadIdx.x;
  poly2 += (params::n / 2) * blockIdx.x + threadIdx.x;

  short2 p1 = *poly1;
  short2 p2 = *poly2;
  short rx = reduce::barrett_reduce(p1.x + p2.x);
  short ry = reduce::barrett_reduce(p1.y + p2.y);
  *poly_out = make_short2(rx, ry);
}

__global__ void add3(short2* poly_out, const short2* poly1, const short2* poly2,
                     const short2* poly3) {
  poly_out += (params::n / 2) * blockIdx.x + threadIdx.x;
  poly1 += (params::n / 2) * blockIdx.x + threadIdx.x;
  poly2 += (params::n / 2) * blockIdx.x + threadIdx.x;
  poly3 += (params::n / 2) * blockIdx.x + threadIdx.x;

  short2 p1 = *poly1;
  short2 p2 = *poly2;
  short2 p3 = *poly3;
  short rx = reduce::barrett_reduce(p1.x + p2.x + p3.x);
  short ry = reduce::barrett_reduce(p1.y + p2.y + p3.y);
  *poly_out = make_short2(rx, ry);
}

__global__ void sub(short2* poly_out, const short2* poly1,
                    const short2* poly2) {
  poly_out += (params::n / 2) * blockIdx.x + threadIdx.x;
  poly1 += (params::n / 2) * blockIdx.x + threadIdx.x;
  poly2 += (params::n / 2) * blockIdx.x + threadIdx.x;

  short2 p1 = *poly1;
  short2 p2 = *poly2;
  short rx = reduce::barrett_reduce(p1.x - p2.x);
  short ry = reduce::barrett_reduce(p1.y - p2.y);
  *poly_out = make_short2(rx, ry);
}

template __global__ void mattimesvec_tomont_plusvec<2>(short2*, const short2*,
                                                       const short2*,
                                                       const short2*);
template __global__ void mattimesvec_tomont_plusvec<3>(short2*, const short2*,
                                                       const short2*,
                                                       const short2*);
template __global__ void mattimesvec_tomont_plusvec<4>(short2*, const short2*,
                                                       const short2*,
                                                       const short2*);

template __global__ void mattimesvec<2>(short2*, const short2*, const short2*);
template __global__ void mattimesvec<3>(short2*, const short2*, const short2*);
template __global__ void mattimesvec<4>(short2*, const short2*, const short2*);

template __global__ void vectimesvec<2>(short2*, const short2*, const short2*);
template __global__ void vectimesvec<3>(short2*, const short2*, const short2*);
template __global__ void vectimesvec<4>(short2*, const short2*, const short2*);

}  // namespace atpqc_cuda::kyber::arithmetic_mt::global
