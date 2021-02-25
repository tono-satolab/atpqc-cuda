//
// device.cuh
// Device functions for generation polynomials.
//
// Copyright (c) 2021 Tatsuki Ono
//
// This software is released under the MIT License.
// https://opensource.org/licenses/mit-license.php
//

#ifndef ATPQC_CUDA_LIB_KYBER_GENPOLY_WARP_DEVICE_CUH_
#define ATPQC_CUDA_LIB_KYBER_GENPOLY_WARP_DEVICE_CUH_

#include <cstddef>
#include <cstdint>

#include "../params.cuh"
#include "../symmetric_ws/device.cuh"

namespace atpqc_cuda::kyber::genpoly_warp::device {

class rej {
 public:
  __device__ unsigned operator()(std::int16_t* poly, unsigned len,
                                 const std::uint8_t* rej_buf,
                                 unsigned buf_len) const;
};

template <unsigned Eta>
class cbd {
 private:
  __device__ std::uint32_t load32_littleendian(const std::uint8_t* x) const;
  __device__ std::uint64_t load48(const std::uint8_t* x) const;

 public:
  __device__ void operator()(short2* poly, const std::uint8_t* cbd_buf) const;
};

__device__ inline unsigned rej::operator()(std::int16_t* poly, unsigned len,
                                           const std::uint8_t* rej_buf,
                                           unsigned buf_len) const {
  unsigned pos = 0;
  unsigned ctr = 0;
  const unsigned shift = 31 - threadIdx.x;
  const unsigned pos_offset = 3 * threadIdx.x;

  while (ctr < len && pos + 3 <= buf_len) {
    bool accept0 = false;
    bool accept1 = false;
    unsigned val0, val1;

    if (pos + pos_offset + 3 <= buf_len) {
      unsigned buf0 = rej_buf[pos + pos_offset + 0];
      unsigned buf1 = rej_buf[pos + pos_offset + 1];
      unsigned buf2 = rej_buf[pos + pos_offset + 2];

      val0 = ((buf0 >> 0) | (buf1 << 8)) & 0xfff;
      val1 = ((buf1 >> 4) | (buf2 << 4)) & 0xfff;

      accept0 = val0 < params::q;
      accept1 = val1 < params::q;
    }

    unsigned warpflag0 = __ballot_sync(0xffffffff, accept0);
    unsigned warpflag1 = __ballot_sync(0xffffffff, accept1);

    unsigned ctr_offset = __popc(warpflag0 << shift) - accept0 +
                          __popc(warpflag1 << shift) - accept1;

    if (accept0 == true && ctr + ctr_offset < len) {
      poly[ctr + ctr_offset] = val0;
      ++ctr_offset;
    }
    if (accept1 == true && ctr + ctr_offset < len) {
      poly[ctr + ctr_offset] = val1;
      ++ctr_offset;
    }

    ctr += __shfl_sync(0xffffffff, ctr_offset, 31);
    pos += 3 * 32;
  }

  return ctr;
}

template <unsigned Eta>
__device__ inline std::uint32_t cbd<Eta>::load32_littleendian(
    const std::uint8_t* x) const {
  std::uint32_t r = static_cast<std::uint32_t>(x[0]) << 0;
  r |= static_cast<std::uint32_t>(x[32]) << 8;
  r |= static_cast<std::uint32_t>(x[64]) << 16;
  r |= static_cast<std::uint32_t>(x[96]) << 24;
  return r;
}

template <unsigned Eta>
__device__ inline std::uint64_t cbd<Eta>::load48(const std::uint8_t* x) const {
  std::uint64_t r = static_cast<std::uint64_t>(x[0]) << 0;
  r |= static_cast<std::uint64_t>(x[1]) << 8;
  r |= static_cast<std::uint64_t>(x[2]) << 16;
  r |= static_cast<std::uint64_t>(x[96]) << 24;
  r |= static_cast<std::uint64_t>(x[97]) << 32;
  r |= static_cast<std::uint64_t>(x[98]) << 40;
  return r;
}

template <>
__device__ inline void cbd<2>::operator()(short2* poly,
                                          const std::uint8_t* cbd_buf) const {
  std::uint32_t t = load32_littleendian(cbd_buf + threadIdx.x);
  std::uint32_t d = t & 0x55555555;
  d += (t >> 1) & 0x55555555;

  unsigned a0 = (d >> 0) & 0x03030303;
  unsigned b0 = (d >> 2) & 0x03030303;
  unsigned a1 = (d >> 4) & 0x03030303;
  unsigned b1 = (d >> 6) & 0x03030303;

  unsigned r0 = __vsub4(a0, b0);
  unsigned r1 = __vsub4(a1, b1);

  poly[threadIdx.x + 0] =
      make_short2(static_cast<std::int16_t>(static_cast<std::int8_t>(r0 >> 0)),
                  static_cast<std::int16_t>(static_cast<std::int8_t>(r1 >> 0)));
  poly[threadIdx.x + 32] =
      make_short2(static_cast<std::int16_t>(static_cast<std::int8_t>(r0 >> 8)),
                  static_cast<std::int16_t>(static_cast<std::int8_t>(r1 >> 8)));
  poly[threadIdx.x + 64] = make_short2(
      static_cast<std::int16_t>(static_cast<std::int8_t>(r0 >> 16)),
      static_cast<std::int16_t>(static_cast<std::int8_t>(r1 >> 16)));
  poly[threadIdx.x + 96] = make_short2(
      static_cast<std::int16_t>(static_cast<std::int8_t>(r0 >> 24)),
      static_cast<std::int16_t>(static_cast<std::int8_t>(r1 >> 24)));
}

template <>
__device__ inline void cbd<3>::operator()(short2* poly,
                                          const std::uint8_t* cbd_buf) const {
  std::uint64_t t = load48(cbd_buf + 3 * threadIdx.x);
  std::uint64_t d = t & 0x00'00'24'92'49'24'92'49;
  d += (t >> 1) & 0x00'00'24'92'49'24'92'49;
  d += (t >> 2) & 0x00'00'24'92'49'24'92'49;

  unsigned a0 = static_cast<std::uint32_t>(d >> 0) & 0x07000007;
  unsigned b0 = static_cast<std::uint32_t>(d >> 3) & 0x07000007;
  unsigned a1 = static_cast<std::uint32_t>(d >> 6) & 0x07000007;
  unsigned b1 = static_cast<std::uint32_t>(d >> 9) & 0x07000007;
  unsigned a2 = static_cast<std::uint32_t>(d >> 12) & 0x07000007;
  unsigned b2 = static_cast<std::uint32_t>(d >> 15) & 0x07000007;
  unsigned a3 = static_cast<std::uint32_t>(d >> 18) & 0x07000007;
  unsigned b3 = static_cast<std::uint32_t>(d >> 21) & 0x07000007;

  unsigned r0 = __vsub4(a0, b0);
  unsigned r1 = __vsub4(a1, b1);
  unsigned r2 = __vsub4(a2, b2);
  unsigned r3 = __vsub4(a3, b3);

  poly[2 * threadIdx.x + 0] =
      make_short2(static_cast<std::int16_t>(static_cast<std::int8_t>(r0 >> 0)),
                  static_cast<std::int16_t>(static_cast<std::int8_t>(r1 >> 0)));
  poly[2 * threadIdx.x + 1] =
      make_short2(static_cast<std::int16_t>(static_cast<std::int8_t>(r2 >> 0)),
                  static_cast<std::int16_t>(static_cast<std::int8_t>(r3 >> 0)));
  poly[2 * threadIdx.x + 64] = make_short2(
      static_cast<std::int16_t>(static_cast<std::int8_t>(r0 >> 24)),
      static_cast<std::int16_t>(static_cast<std::int8_t>(r1 >> 24)));
  poly[2 * threadIdx.x + 65] = make_short2(
      static_cast<std::int16_t>(static_cast<std::int8_t>(r2 >> 24)),
      static_cast<std::int16_t>(static_cast<std::int8_t>(r3 >> 24)));
}

}  // namespace atpqc_cuda::kyber::genpoly_warp::device

#endif
