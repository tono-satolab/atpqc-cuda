//
// global.cuh
// Kernel header of NTT (128 threads version).
//
// Copyright (c) 2021 Tatsuki Ono
//
// This software is released under the MIT License.
// https://opensource.org/licenses/mit-license.php
//

#ifndef ATPQC_CUDA_LIB_KYBER_NTT_CTGS_128T_GLOBAL_CUH_
#define ATPQC_CUDA_LIB_KYBER_NTT_CTGS_128T_GLOBAL_CUH_

namespace atpqc_cuda::kyber::ntt_ctgs_128t::global {

__global__ void fwdntt(short2* poly, unsigned npolys);
__global__ void invntt_tomont(short2* poly, unsigned npolys);

}  // namespace atpqc_cuda::kyber::ntt_ctgs_128t::global

#endif
