//
// device.cuh
// Device function header for NTT (128 threads version).
//
// Copyright (c) 2021 Tatsuki Ono
//
// This software is released under the MIT License.
// https://opensource.org/licenses/mit-license.php
//

#ifndef ATPQC_CUDA_LIB_KYBER_NTT_CTGS_128T_DEVICE_CUH_
#define ATPQC_CUDA_LIB_KYBER_NTT_CTGS_128T_DEVICE_CUH_

#include <cstdint>

namespace atpqc_cuda::kyber::ntt_ctgs_128t::device {

class fwdntt {
 public:
  __device__ void operator()(std::int16_t* shared_poly) const;
};

class invntt {
 public:
  __device__ void operator()(std::int16_t* shared_poly) const;
};

}  // namespace atpqc_cuda::kyber::ntt_ctgs_128t::device

#endif
