//
// kernel_params.cuh
// Parameters for kernels of generation polynomials.
//
// Copyright (c) 2021 Tatsuki Ono
//
// This software is released under the MIT License.
// https://opensource.org/licenses/mit-license.php
//

#ifndef ATPQC_CUDA_LIB_KYBER_GENPOLY_WARP_KERNEL_PARAMS_CUH_
#define ATPQC_CUDA_LIB_KYBER_GENPOLY_WARP_KERNEL_PARAMS_CUH_

#include <algorithm>
#include <cstdint>

#include "../params.cuh"
#include "../symmetric_ws/device.cuh"

namespace atpqc_cuda::kyber::genpoly_warp::kernel_params {

struct genmatrix {
  static constexpr unsigned poly_bytes = sizeof(short2) * params::n / 2;
  static constexpr unsigned extseed_bytes = params::symbytes + 2;
  static constexpr unsigned xof_blockbytes =
      symmetric_ws::device::xof::blockbytes;
  static constexpr unsigned xof_nblocks =
      (12 * params::n / 8 * (1 << 12) / params::q + xof_blockbytes) /
      xof_blockbytes;
  static constexpr unsigned rej_bytes = xof_blockbytes * xof_nblocks;
  static constexpr unsigned bytes_buflen =
      (std::max(extseed_bytes, rej_bytes) + 4) / 4 * 4;
  static constexpr unsigned smem_byte_per_warp = poly_bytes + bytes_buflen;
};

template <unsigned Eta>
struct gennoise {
  static constexpr unsigned eta = Eta;
  static constexpr unsigned extseed_bytes = params::symbytes + 1;
  static constexpr unsigned cbd_bytes = eta * params::n / 4;
  static constexpr unsigned prf_blockbytes = symmetric_ws::device::prf::rate;
  static constexpr unsigned prf_nblocks =
      symmetric_ws::device::prf::len_to_nblocks(cbd_bytes);
  static constexpr unsigned buflen =
      (std::max(extseed_bytes, prf_blockbytes* prf_nblocks) + 4) / 4 * 4;
  static constexpr unsigned smem_byte_per_warp = buflen;
};

}  // namespace atpqc_cuda::kyber::genpoly_warp::kernel_params

#endif
