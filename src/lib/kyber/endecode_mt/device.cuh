//
// device.cuh
// Device functions for encoding/decoding polynomials.
//
// Copyright (c) 2021 Tatsuki Ono
//
// This software is released under the MIT License.
// https://opensource.org/licenses/mit-license.php
//

#ifndef ATPQC_CUDA_LIB_KYBER_ENDECODE_MT_DEVICE_CUH_
#define ATPQC_CUDA_LIB_KYBER_ENDECODE_MT_DEVICE_CUH_

#include <cstdint>

#include "../reduce.cuh"

namespace atpqc_cuda::kyber::endecode_mt::device {

__device__ inline std::uint16_t psr(std::int16_t a) noexcept {
  // map to positive standard representatives

  a += (a >> 15) & params::q;
  return a;
}

template <unsigned Dv>
class poly_compress {
 public:
  __device__ void operator()(std::uint8_t* cbytes, const short2* coeffs) const;
};

template <unsigned Dv>
class poly_decompress {
 public:
  __device__ void operator()(short2* coeffs, const std::uint8_t* cbytes) const;
};

template <unsigned Du>
class polyvec_compress {
 public:
  __device__ void operator()(std::uint8_t* cbytes, const short2* coeffs) const;
};

template <unsigned Du>
class polyvec_decompress {
 public:
  __device__ void operator()(short2* coeffs, const std::uint8_t* cbytes) const;
};

class polyvec_tobytes {
 public:
  __device__ void operator()(std::uint8_t* bytes, const short2* coeffs) const;
};

class polyvec_frombytes {
 public:
  __device__ void operator()(short2* coeffs, const std::uint8_t* bytes) const;
};

class poly_frommsg {
 public:
  __device__ void operator()(short2* coeffs, const std::uint8_t* msg) const;
};

class poly_tomsg {
 public:
  __device__ void operator()(std::uint8_t* msg, const short2* coeffs) const;
};

template <>
__device__ inline void poly_compress<4>::operator()(
    std::uint8_t* cbytes, const short2* coeffs) const {
  auto compress_unit = [](std::int16_t x) noexcept -> std::uint8_t {
    return (((static_cast<std::uint16_t>(x) << 4) + params::q / 2) /
            params::q) &
           0xf;
  };

  short2 c0 = coeffs[0];

  std::uint8_t t0 = compress_unit(psr(c0.x));
  std::uint8_t t1 = compress_unit(psr(c0.y));

  cbytes[0] = t0 | (t1 << 4);
}

template <>
__device__ inline void poly_compress<5>::operator()(
    std::uint8_t* cbytes, const short2* coeffs) const {
  auto compress_unit = [](std::int16_t x) noexcept -> std::uint8_t {
    return (((static_cast<std::uint16_t>(x) << 5) + params::q / 2) /
            params::q) &
           0x1f;
  };

  short2 c0 = coeffs[0];
  short2 c1 = coeffs[1];
  short2 c2 = coeffs[2];
  short2 c3 = coeffs[3];

  std::uint8_t t0 = compress_unit(psr(c0.x));
  std::uint8_t t1 = compress_unit(psr(c0.y));
  std::uint8_t t2 = compress_unit(psr(c1.x));
  std::uint8_t t3 = compress_unit(psr(c1.y));
  std::uint8_t t4 = compress_unit(psr(c2.x));
  std::uint8_t t5 = compress_unit(psr(c2.y));
  std::uint8_t t6 = compress_unit(psr(c3.x));
  std::uint8_t t7 = compress_unit(psr(c3.y));

  cbytes[0] = (t0 >> 0) | (t1 << 5);
  cbytes[1] = (t1 >> 3) | (t2 << 2) | (t3 << 7);
  cbytes[2] = (t3 >> 1) | (t4 << 4);
  cbytes[3] = (t4 >> 4) | (t5 << 1) | (t6 << 6);
  cbytes[4] = (t6 >> 2) | (t7 << 3);
}

template <>
__device__ inline void poly_decompress<4>::operator()(
    short2* coeffs, const std::uint8_t* cbytes) const {
  auto decompress_unit = [](std::uint8_t x) noexcept -> std::int16_t {
    return (static_cast<std::uint16_t>(x) * params::q + 8) >> 4;
  };

  std::uint8_t a0 = cbytes[0];

  std::int16_t t0 = decompress_unit(a0 & 0b1111);
  std::int16_t t1 = decompress_unit(a0 >> 4);

  coeffs[0] = make_short2(t0, t1);
}

template <>
__device__ inline void poly_decompress<5>::operator()(
    short2* coeffs, const std::uint8_t* cbytes) const {
  auto decompress_unit = [](std::uint8_t x) noexcept -> std::int16_t {
    return (static_cast<std::uint32_t>(x) * params::q + 16) >> 5;
  };

  std::uint8_t a0 = cbytes[0];
  std::uint8_t a1 = cbytes[1];
  std::uint8_t a2 = cbytes[2];
  std::uint8_t a3 = cbytes[3];
  std::uint8_t a4 = cbytes[4];

  std::int16_t t0 = decompress_unit(((a0 >> 0)) & 0b11111);
  std::int16_t t1 = decompress_unit(((a0 >> 5) | (a1 << 3)) & 0b11111);
  std::int16_t t2 = decompress_unit(((a1 >> 2)) & 0b11111);
  std::int16_t t3 = decompress_unit(((a1 >> 7) | (a2 << 1)) & 0b11111);
  std::int16_t t4 = decompress_unit(((a2 >> 4) | (a3 << 4)) & 0b11111);
  std::int16_t t5 = decompress_unit(((a3 >> 1)) & 0b11111);
  std::int16_t t6 = decompress_unit(((a3 >> 6) | (a4 << 2)) & 0b11111);
  std::int16_t t7 = decompress_unit(((a4 >> 3)));

  coeffs[0] = make_short2(t0, t1);
  coeffs[1] = make_short2(t2, t3);
  coeffs[2] = make_short2(t4, t5);
  coeffs[3] = make_short2(t6, t7);
}

template <>
__device__ inline void polyvec_compress<10>::operator()(
    std::uint8_t* cbytes, const short2* coeffs) const {
  auto compress_unit = [](std::int16_t x) noexcept -> std::uint16_t {
    return (((static_cast<std::uint32_t>(x) << 10) + params::q / 2) /
            params::q) &
           0x3ff;
  };

  short2 c0 = coeffs[0];
  short2 c1 = coeffs[1];

  std::uint16_t t0 = compress_unit(psr(c0.x));
  std::uint16_t t1 = compress_unit(psr(c0.y));
  std::uint16_t t2 = compress_unit(psr(c1.x));
  std::uint16_t t3 = compress_unit(psr(c1.y));

  cbytes[0] = (t0 >> 0);
  cbytes[1] = (t0 >> 8) | (t1 << 2);
  cbytes[2] = (t1 >> 6) | (t2 << 4);
  cbytes[3] = (t2 >> 4) | (t3 << 6);
  cbytes[4] = (t3 >> 2);
}

template <>
__device__ inline void polyvec_compress<11>::operator()(
    std::uint8_t* cbytes, const short2* coeffs) const {
  auto compress_unit = [](std::int16_t x) noexcept -> std::uint16_t {
    return (((static_cast<std::uint32_t>(x) << 11) + params::q / 2) /
            params::q) &
           0x7ff;
  };

  short2 c0 = coeffs[0];
  short2 c1 = coeffs[1];
  short2 c2 = coeffs[2];
  short2 c3 = coeffs[3];

  std::uint16_t t0 = compress_unit(psr(c0.x));
  std::uint16_t t1 = compress_unit(psr(c0.y));
  std::uint16_t t2 = compress_unit(psr(c1.x));
  std::uint16_t t3 = compress_unit(psr(c1.y));
  std::uint16_t t4 = compress_unit(psr(c2.x));
  std::uint16_t t5 = compress_unit(psr(c2.y));
  std::uint16_t t6 = compress_unit(psr(c3.x));
  std::uint16_t t7 = compress_unit(psr(c3.y));

  cbytes[0] = (t0 >> 0);
  cbytes[1] = (t0 >> 8) | (t1 << 3);
  cbytes[2] = (t1 >> 5) | (t2 << 6);
  cbytes[3] = (t2 >> 2);
  cbytes[4] = (t2 >> 10) | (t3 << 1);
  cbytes[5] = (t3 >> 7) | (t4 << 4);
  cbytes[6] = (t4 >> 4) | (t5 << 7);
  cbytes[7] = (t5 >> 1);
  cbytes[8] = (t5 >> 9) | (t6 << 2);
  cbytes[9] = (t6 >> 6) | (t7 << 5);
  cbytes[10] = (t7 >> 3);
}

template <>
__device__ inline void polyvec_decompress<10>::operator()(
    short2* coeffs, const std::uint8_t* cbytes) const {
  auto decompress_unit = [](std::uint16_t x) noexcept -> std::int16_t {
    return (static_cast<std::uint32_t>(x) * params::q + 512) >> 10;
  };

  std::uint8_t a0 = cbytes[0];
  std::uint8_t a1 = cbytes[1];
  std::uint8_t a2 = cbytes[2];
  std::uint8_t a3 = cbytes[3];
  std::uint8_t a4 = cbytes[4];

  std::int16_t t0 = decompress_unit(
      ((a0 >> 0) | (static_cast<std::uint16_t>(a1) << 8)) & 0x3ff);
  std::int16_t t1 = decompress_unit(
      ((a1 >> 2) | (static_cast<std::uint16_t>(a2) << 6)) & 0x3ff);
  std::int16_t t2 = decompress_unit(
      ((a2 >> 4) | (static_cast<std::uint16_t>(a3) << 4)) & 0x3ff);
  std::int16_t t3 = decompress_unit(
      ((a3 >> 6) | (static_cast<std::uint16_t>(a4) << 2)) & 0x3ff);

  coeffs[0] = make_short2(t0, t1);
  coeffs[1] = make_short2(t2, t3);
}

template <>
__device__ inline void polyvec_decompress<11>::operator()(
    short2* coeffs, const std::uint8_t* cbytes) const {
  auto decompress_unit = [](std::uint16_t x) noexcept -> std::int16_t {
    return (static_cast<std::uint32_t>(x) * params::q + 1024) >> 11;
  };

  std::uint8_t a0 = cbytes[0];
  std::uint8_t a1 = cbytes[1];
  std::uint8_t a2 = cbytes[2];
  std::uint8_t a3 = cbytes[3];
  std::uint8_t a4 = cbytes[4];
  std::uint8_t a5 = cbytes[5];
  std::uint8_t a6 = cbytes[6];
  std::uint8_t a7 = cbytes[7];
  std::uint8_t a8 = cbytes[8];
  std::uint8_t a9 = cbytes[9];
  std::uint8_t a10 = cbytes[10];

  std::int16_t t0 = decompress_unit(
      ((a0 >> 0) | (static_cast<std::uint16_t>(a1) << 8)) & 0x7ff);
  std::int16_t t1 = decompress_unit(
      ((a1 >> 3) | (static_cast<std::uint16_t>(a2) << 5)) & 0x7ff);
  std::int16_t t2 =
      decompress_unit(((a2 >> 6) | (static_cast<std::uint16_t>(a3) << 2) |
                       (static_cast<std::uint16_t>(a4) << 10)) &
                      0x7ff);
  std::int16_t t3 = decompress_unit(
      ((a4 >> 1) | (static_cast<std::uint16_t>(a5) << 7)) & 0x7ff);
  std::int16_t t4 = decompress_unit(
      ((a5 >> 4) | (static_cast<std::uint16_t>(a6) << 4)) & 0x7ff);
  std::int16_t t5 =
      decompress_unit(((a6 >> 7) | (static_cast<std::uint16_t>(a7) << 1) |
                       (static_cast<std::uint16_t>(a8) << 9)) &
                      0x7ff);
  std::int16_t t6 = decompress_unit(
      ((a8 >> 2) | (static_cast<std::uint16_t>(a9) << 6)) & 0x7ff);
  std::int16_t t7 = decompress_unit(
      ((a9 >> 5) | (static_cast<std::uint16_t>(a10) << 3)) & 0x7ff);

  coeffs[0] = make_short2(t0, t1);
  coeffs[1] = make_short2(t2, t3);
  coeffs[2] = make_short2(t4, t5);
  coeffs[3] = make_short2(t6, t7);
}

__device__ inline void polyvec_tobytes::operator()(std::uint8_t* bytes,
                                                   const short2* coeffs) const {
  short2 c0 = coeffs[0];

  std::uint16_t t0 = psr(c0.x);
  std::uint16_t t1 = psr(c0.y);

  bytes[0] = (t0 >> 0);
  bytes[1] = (t0 >> 8) | (t1 << 4);
  bytes[2] = (t1 >> 4);
}

__device__ inline void polyvec_frombytes::operator()(
    short2* coeffs, const std::uint8_t* bytes) const {
  std::uint16_t a0 = bytes[0];
  std::uint16_t a1 = bytes[1];
  std::uint16_t a2 = bytes[2];

  std::int16_t t0 = ((a0 >> 0) | (a1 << 8)) & 0xfff;
  std::int16_t t1 = ((a1 >> 4) | (a2 << 4)) & 0xfff;

  coeffs[0] = make_short2(t0, t1);
}

__device__ inline void poly_frommsg::operator()(short2* coeffs,
                                                const std::uint8_t* msg) const {
  constexpr std::int16_t val = (params::q + 1) / 2;

  std::uint8_t m = *msg;

  std::int16_t t0 = (-static_cast<std::int16_t>((m >> 0) & 0b1)) & val;
  std::int16_t t1 = (-static_cast<std::int16_t>((m >> 1) & 0b1)) & val;
  std::int16_t t2 = (-static_cast<std::int16_t>((m >> 2) & 0b1)) & val;
  std::int16_t t3 = (-static_cast<std::int16_t>((m >> 3) & 0b1)) & val;
  std::int16_t t4 = (-static_cast<std::int16_t>((m >> 4) & 0b1)) & val;
  std::int16_t t5 = (-static_cast<std::int16_t>((m >> 5) & 0b1)) & val;
  std::int16_t t6 = (-static_cast<std::int16_t>((m >> 6) & 0b1)) & val;
  std::int16_t t7 = (-static_cast<std::int16_t>((m >> 7) & 0b1)) & val;

  coeffs[0] = make_short2(t0, t1);
  coeffs[1] = make_short2(t2, t3);
  coeffs[2] = make_short2(t4, t5);
  coeffs[3] = make_short2(t6, t7);
}

__device__ inline void poly_tomsg::operator()(std::uint8_t* msg,
                                              const short2* coeffs) const {
  auto compress_unit = [](std::int16_t x) noexcept -> std::uint8_t {
    return (((static_cast<std::uint16_t>(x) << 1) + params::q / 2) /
            params::q) &
           0b1;
  };

  short2 c0 = coeffs[0];
  short2 c1 = coeffs[1];
  short2 c2 = coeffs[2];
  short2 c3 = coeffs[3];

  *msg = compress_unit(psr(c0.x)) << 0 | compress_unit(psr(c0.y)) << 1 |
         compress_unit(psr(c1.x)) << 2 | compress_unit(psr(c1.y)) << 3 |
         compress_unit(psr(c2.x)) << 4 | compress_unit(psr(c2.y)) << 5 |
         compress_unit(psr(c3.x)) << 6 | compress_unit(psr(c3.y)) << 7;
}

}  // namespace atpqc_cuda::kyber::endecode_mt::device

#endif
