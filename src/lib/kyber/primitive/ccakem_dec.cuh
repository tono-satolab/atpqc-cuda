//
// ccakem_dec.cuh
// Host function of Kyber.Decaps.
//
// Copyright (c) 2021 Tatsuki Ono
//
// This software is released under the MIT License.
// https://opensource.org/licenses/mit-license.php
//

#ifndef ATPQC_CUDA_LIB_KYBER_PRIMITIVE_CCAKEM_DEC_CUH_
#define ATPQC_CUDA_LIB_KYBER_PRIMITIVE_CCAKEM_DEC_CUH_

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "../../cuda_debug.hpp"
#include "../../cuda_resource.hpp"
#include "../params.cuh"
#include "common.cuh"
#include "cpapke_dec.cuh"
#include "cpapke_enc.cuh"

namespace atpqc_cuda::kyber::primitive::ccakem_dec {

template <class Variant>
struct mem_resource {
  using variant = Variant;
  cuda_resource::device_pitched_memory<std::uint8_t> buf;
  cuda_resource::device_pitched_memory<std::uint8_t> kr;
  cuda_resource::device_pitched_memory<std::uint8_t> cmp;

  cpapke_enc::mem_resource<variant> pke_enc_mr;
  cpapke_dec::mem_resource<variant> pke_dec_mr;

  mem_resource() = delete;
  mem_resource(unsigned ninputs)
      : buf(2 * params::symbytes, ninputs),
        kr(2 * params::symbytes, ninputs),
        cmp(params::ciphertextbytes<variant>, ninputs),
        pke_enc_mr(ninputs),
        pke_dec_mr(ninputs) {}
};

template <class Variant, class CpapkeEnc, class CpapkeDec, class HashH,
          class HashG, class Kdf, class VerifyCmov>
class dec {
 private:
  using variant = Variant;

  unsigned nin;
  CpapkeEnc pke_enc;
  CpapkeDec pke_dec;
  HashH hash_ct;
  HashG hash_coin;
  Kdf kdf;
  VerifyCmov verify_cmov;

 public:
  struct graph_args {
    using pke_enc_args_type = typename CpapkeEnc::graph_args;
    using pke_dec_args_type = typename CpapkeDec::graph_args;
    using hash_ct_args_type = typename HashH::args_type;
    using hash_coin_args_type = typename HashG::args_type;
    using kdf_args_type = typename Kdf::args_type;
    using verify_cmov_args_type = typename VerifyCmov::args_type;

    pke_enc_args_type pke_enc_args;
    pke_dec_args_type pke_dec_args;

    std::unique_ptr<hash_ct_args_type> hash_ct_args;
    std::unique_ptr<hash_coin_args_type> hash_coin_args;
    std::unique_ptr<kdf_args_type> kdf_args;
    std::unique_ptr<verify_cmov_args_type> verify_cmov_args;
  };

  graph_args join_graph(cudaGraph_t graph, std::uint8_t* ss,
                        std::size_t ss_pitch, cudaGraphNode_t ss_empty,
                        cudaGraphNode_t* ss_available_ptr,
                        const std::uint8_t* ct, std::size_t ct_pitch,
                        cudaGraphNode_t ct_ready, cudaGraphNode_t* ct_used_ptr,
                        const std::uint8_t* sk, std::size_t sk_pitch,
                        cudaGraphNode_t sk_ready, cudaGraphNode_t* sk_used_ptr,
                        const mem_resource<variant>& mr) const {
    graph_args args_pack;
    std::uint8_t* buf_ptr = mr.buf.get_ptr();
    std::size_t buf_pitch = mr.buf.get_pitch();
    std::uint8_t* kr_ptr = mr.kr.get_ptr();
    std::size_t kr_pitch = mr.kr.get_pitch();
    std::uint8_t* cmp_ptr = mr.cmp.get_ptr();
    std::size_t cmp_pitch = mr.cmp.get_pitch();
    const std::uint8_t* pk_in_sk = sk + params::indcpa_secretkeybytes<variant>;
    const std::uint8_t* z_in_sk =
        sk + params::secretkeybytes<variant> - params::symbytes;

    cudaGraphNode_t empty_root_node;
    { CCC(cudaGraphAddEmptyNode(&empty_root_node, graph, nullptr, 0)); }

    cudaGraphNode_t pke_dec_m_node;
    cudaGraphNode_t pke_dec_c_node;
    cudaGraphNode_t pke_dec_sk_node;
    {
      args_pack.pke_dec_args = pke_dec.join_graph(
          graph, buf_ptr, buf_pitch, empty_root_node, &pke_dec_m_node, ct,
          ct_pitch, ct_ready, &pke_dec_c_node, sk, sk_pitch, sk_ready,
          &pke_dec_sk_node, mr.pke_dec_mr);
    }

    cudaGraphNode_t cpyh_node;
    {
      cudaMemcpy3DParms memcpy_params;
      memcpy_params.srcPos = make_cudaPos(
          params::secretkeybytes<variant> - 2 * params::symbytes, 0, 0);
      memcpy_params.srcPtr =
          make_cudaPitchedPtr(const_cast<std::uint8_t*>(sk), sk_pitch,
                              params::secretkeybytes<variant>, nin);
      memcpy_params.srcArray = nullptr;
      memcpy_params.dstPos = make_cudaPos(params::symbytes, 0, 0);
      memcpy_params.dstPtr =
          make_cudaPitchedPtr(buf_ptr, buf_pitch, 2 * params::symbytes, nin);
      memcpy_params.dstArray = nullptr;
      memcpy_params.extent = make_cudaExtent(params::symbytes, nin, 1);
      memcpy_params.kind = cudaMemcpyDeviceToDevice;
      std::array dep_array{sk_ready};
      CCC(cudaGraphAddMemcpyNode(&cpyh_node, graph, dep_array.data(),
                                 dep_array.size(), &memcpy_params));
    }

    cudaGraphNode_t hashcoin_node;
    {
      args_pack.hash_coin_args = hash_coin.generate_args(
          kr_ptr, kr_pitch, buf_ptr, buf_pitch, 2 * params::symbytes);
      cudaKernelNodeParams kernel_params;
      kernel_params.func = hash_coin.get_func();
      kernel_params.gridDim = hash_coin.get_grid_dim();
      kernel_params.blockDim = hash_coin.get_block_dim();
      kernel_params.sharedMemBytes = hash_coin.get_shared_bytes();
      kernel_params.kernelParams = args_pack.hash_coin_args->get_args_ptr();
      kernel_params.extra = nullptr;
      std::array dep_array{pke_dec_m_node, cpyh_node};
      CCC(cudaGraphAddKernelNode(&hashcoin_node, graph, dep_array.data(),
                                 dep_array.size(), &kernel_params));
    }

    cudaGraphNode_t pke_enc_c_node;
    cudaGraphNode_t pke_enc_m_node;
    cudaGraphNode_t pke_enc_pk_node;
    cudaGraphNode_t pke_enc_coin_node;
    {
      args_pack.pke_enc_args = pke_enc.join_graph(
          graph, cmp_ptr, cmp_pitch, empty_root_node, &pke_enc_c_node, buf_ptr,
          buf_pitch, pke_dec_m_node, &pke_enc_m_node, pk_in_sk, sk_pitch,
          sk_ready, &pke_enc_pk_node, kr_ptr + params::symbytes, kr_pitch,
          hashcoin_node, &pke_enc_coin_node, mr.pke_enc_mr);
    }

    cudaGraphNode_t hashct_node;
    {
      args_pack.hash_ct_args =
          hash_ct.generate_args(kr_ptr + params::symbytes, kr_pitch, ct,
                                ct_pitch, params::ciphertextbytes<variant>);
      cudaKernelNodeParams kernel_params;
      kernel_params.func = hash_ct.get_func();
      kernel_params.gridDim = hash_ct.get_grid_dim();
      kernel_params.blockDim = hash_ct.get_block_dim();
      kernel_params.sharedMemBytes = hash_ct.get_shared_bytes();
      kernel_params.kernelParams = args_pack.hash_ct_args->get_args_ptr();
      kernel_params.extra = nullptr;
      std::array dep_array{ct_ready, pke_enc_coin_node};
      CCC(cudaGraphAddKernelNode(&hashct_node, graph, dep_array.data(),
                                 dep_array.size(), &kernel_params));
    }

    cudaGraphNode_t verify_cmov_node;
    {
      args_pack.verify_cmov_args = verify_cmov.generate_args(
          kr_ptr, kr_pitch, z_in_sk, sk_pitch, params::symbytes, ct, ct_pitch,
          cmp_ptr, cmp_pitch, params::ciphertextbytes<variant>);
      cudaKernelNodeParams kernel_params;
      kernel_params.func = verify_cmov.get_func();
      kernel_params.gridDim = verify_cmov.get_grid_dim();
      kernel_params.blockDim = verify_cmov.get_block_dim();
      kernel_params.sharedMemBytes = verify_cmov.get_shared_bytes();
      kernel_params.kernelParams = args_pack.verify_cmov_args->get_args_ptr();
      kernel_params.extra = nullptr;
      std::array dep_array{hashcoin_node, sk_ready, ct_ready, pke_enc_c_node};
      CCC(cudaGraphAddKernelNode(&verify_cmov_node, graph, dep_array.data(),
                                 dep_array.size(), &kernel_params));
    }

    cudaGraphNode_t kdf_node;
    {
      args_pack.kdf_args = kdf.generate_args(ss, ss_pitch, kr_ptr, kr_pitch,
                                             2 * params::symbytes);
      cudaKernelNodeParams kernel_params;
      kernel_params.func = kdf.get_func();
      kernel_params.gridDim = kdf.get_grid_dim();
      kernel_params.blockDim = kdf.get_block_dim();
      kernel_params.sharedMemBytes = kdf.get_shared_bytes();
      kernel_params.kernelParams = args_pack.kdf_args->get_args_ptr();
      kernel_params.extra = nullptr;
      std::array dep_array{ss_empty, hashct_node, verify_cmov_node};
      CCC(cudaGraphAddKernelNode(&kdf_node, graph, dep_array.data(),
                                 dep_array.size(), &kernel_params));
    }

    *ss_available_ptr = kdf_node;

    {
      std::array dep_array{pke_dec_c_node, hashct_node, verify_cmov_node};
      CCC(cudaGraphAddEmptyNode(ct_used_ptr, graph, dep_array.data(),
                                dep_array.size()));
    }

    {
      std::array dep_array{pke_dec_sk_node, pke_enc_pk_node, cpyh_node,
                           verify_cmov_node};
      CCC(cudaGraphAddEmptyNode(sk_used_ptr, graph, dep_array.data(),
                                dep_array.size()));
    }

    return args_pack;
  }

  dec(unsigned ninputs, Variant variant_v, const CpapkeEnc& cpe,
      const CpapkeDec& cpd, const HashH& hash_h, const HashG& hash_g,
      const Kdf& hash_kdf, const VerifyCmov& vc)
      : nin(ninputs),
        pke_enc(cpe),
        pke_dec(cpd),
        hash_ct(hash_h),
        hash_coin(hash_g),
        kdf(hash_kdf),
        verify_cmov(vc) {}
};

}  // namespace atpqc_cuda::kyber::primitive::ccakem_dec

#endif
