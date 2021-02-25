//
// global.cu
// Kernels for encoding/decoding polynomials.
//
// Copyright (c) 2021 Tatsuki Ono
//
// This software is released under the MIT License.
// https://opensource.org/licenses/mit-license.php
//

#include "device.cuh"
#include "global.cuh"
#include "kernel_params.cuh"

namespace atpqc_cuda::kyber::endecode_mt::global {

template <unsigned Dv>
__global__ void poly_compress(std::uint8_t* cbytes, std::size_t cbytes_pitch,
                              const short2* poly, unsigned ninputs) {
  constexpr unsigned dv = Dv;
  using kp = kernel_params::poly_de_compress<dv>;
  device::poly_compress<dv> compress;

  if (unsigned pos = blockIdx.x * blockDim.y + threadIdx.y; pos < ninputs) {
    cbytes += cbytes_pitch * pos + kp::cbyte_per_thread * threadIdx.x;
    poly += (params::n / 2) * pos + kp::coeff_per_thread * threadIdx.x;

    compress(cbytes, poly);
  }
}

template <unsigned Dv>
__global__ void poly_decompress(short2* poly, const std::uint8_t* cbytes,
                                std::size_t cbytes_pitch, unsigned ninputs) {
  constexpr unsigned dv = Dv;
  using kp = kernel_params::poly_de_compress<dv>;
  device::poly_decompress<dv> decompress;

  if (unsigned pos = blockIdx.x * blockDim.y + threadIdx.y; pos < ninputs) {
    poly += (params::n / 2) * pos + kp::coeff_per_thread * threadIdx.x;
    cbytes += cbytes_pitch * pos + kp::cbyte_per_thread * threadIdx.x;

    decompress(poly, cbytes);
  }
}

template <unsigned K, unsigned Du>
__global__ void polyvec_compress(std::uint8_t* cbytes, std::size_t cbytes_pitch,
                                 const short2* polyvec, unsigned ninputs) {
  constexpr unsigned k = K;
  constexpr unsigned du = Du;
  using kp = kernel_params::polyvec_de_compress<k, du>;
  device::polyvec_compress<du> compress;

  if (unsigned pos = blockIdx.x * blockDim.y + threadIdx.y; pos < ninputs) {
    cbytes += cbytes_pitch * pos + kp::cbyte_per_thread * threadIdx.x;
    polyvec += (k * params::n / 2) * pos + kp::coeff_per_thread * threadIdx.x;

    compress(cbytes, polyvec);
  }
}

template <unsigned K, unsigned Du>
__global__ void polyvec_decompress(short2* polyvec, const std::uint8_t* cbytes,
                                   std::size_t cbytes_pitch, unsigned ninputs) {
  constexpr unsigned k = K;
  constexpr unsigned du = Du;
  using kp = kernel_params::polyvec_de_compress<k, du>;
  device::polyvec_decompress<du> decompress;

  if (unsigned pos = blockIdx.x * blockDim.y + threadIdx.y; pos < ninputs) {
    polyvec += (k * params::n / 2) * pos + kp::coeff_per_thread * threadIdx.x;
    cbytes += cbytes_pitch * pos + kp::cbyte_per_thread * threadIdx.x;

    decompress(polyvec, cbytes);
  }
}

template <unsigned K>
__global__ void polyvec_tobytes(std::uint8_t* bytes, std::size_t bytes_pitch,
                                const short2* polyvec, unsigned ninputs) {
  constexpr unsigned k = K;
  using kp = kernel_params::polyvec_bytes<k>;
  device::polyvec_tobytes tobytes;

  if (unsigned pos = blockIdx.x * blockDim.y + threadIdx.y; pos < ninputs) {
    bytes += bytes_pitch * pos + kp::byte_per_thread * threadIdx.x;
    polyvec += (k * params::n / 2) * pos + kp::coeff_per_thread * threadIdx.x;

    tobytes(bytes, polyvec);
  }
}

template <unsigned K>
__global__ void polyvec_frombytes(short2* polyvec, const std::uint8_t* bytes,
                                  std::size_t bytes_pitch, unsigned ninputs) {
  constexpr unsigned k = K;
  using kp = kernel_params::polyvec_bytes<K>;
  device::polyvec_frombytes frombytes;

  if (unsigned pos = blockIdx.x * blockDim.y + threadIdx.y; pos < ninputs) {
    polyvec += (k * params::n / 2) * pos + kp::coeff_per_thread * threadIdx.x;
    bytes += bytes_pitch * pos + kp::byte_per_thread * threadIdx.x;

    frombytes(polyvec, bytes);
  }
}

__global__ void poly_frommsg(short2* poly, const std::uint8_t* msg,
                             std::size_t msg_pitch, unsigned ninputs) {
  using kp = kernel_params::poly_msg;
  device::poly_frommsg frommsg;

  if (unsigned pos = blockIdx.x * blockDim.y + threadIdx.y; pos < ninputs) {
    poly += (params::n / 2) * pos + kp::coeff_per_thread * threadIdx.x;
    msg += msg_pitch * pos + kp::msgbyte_per_thread * threadIdx.x;

    frommsg(poly, msg);
  }
}

__global__ void poly_tomsg(std::uint8_t* msg, std::size_t msg_pitch,
                           const short2* poly, unsigned ninputs) {
  using kp = kernel_params::poly_msg;
  device::poly_tomsg tomsg;

  if (unsigned pos = blockIdx.x * blockDim.y + threadIdx.y; pos < ninputs) {
    msg += msg_pitch * pos + kp::msgbyte_per_thread * threadIdx.x;
    poly += (params::n / 2) * pos + kp::coeff_per_thread * threadIdx.x;

    tomsg(msg, poly);
  }
}

template __global__ void poly_compress<4>(std::uint8_t*, std::size_t,
                                          const short2*, unsigned);
template __global__ void poly_compress<5>(std::uint8_t*, std::size_t,
                                          const short2*, unsigned);
template __global__ void poly_decompress<4>(short2*, const std::uint8_t*,
                                            std::size_t, unsigned);
template __global__ void poly_decompress<5>(short2*, const std::uint8_t*,
                                            std::size_t, unsigned);
template __global__ void polyvec_compress<2, 10>(std::uint8_t*, std::size_t,
                                                 const short2*, unsigned);
template __global__ void polyvec_compress<3, 10>(std::uint8_t*, std::size_t,
                                                 const short2*, unsigned);
template __global__ void polyvec_compress<4, 11>(std::uint8_t*, std::size_t,
                                                 const short2*, unsigned);
template __global__ void polyvec_decompress<2, 10>(short2*, const std::uint8_t*,
                                                   std::size_t, unsigned);
template __global__ void polyvec_decompress<3, 10>(short2*, const std::uint8_t*,
                                                   std::size_t, unsigned);
template __global__ void polyvec_decompress<4, 11>(short2*, const std::uint8_t*,
                                                   std::size_t, unsigned);
template __global__ void polyvec_tobytes<2>(std::uint8_t*, std::size_t,
                                            const short2*, unsigned);
template __global__ void polyvec_tobytes<3>(std::uint8_t*, std::size_t,
                                            const short2*, unsigned);
template __global__ void polyvec_tobytes<4>(std::uint8_t*, std::size_t,
                                            const short2*, unsigned);
template __global__ void polyvec_frombytes<2>(short2*, const std::uint8_t*,
                                              std::size_t, unsigned);
template __global__ void polyvec_frombytes<3>(short2*, const std::uint8_t*,
                                              std::size_t, unsigned);
template __global__ void polyvec_frombytes<4>(short2*, const std::uint8_t*,
                                              std::size_t, unsigned);

}  // namespace atpqc_cuda::kyber::endecode_mt::global
