//
// ccakem_enc.cuh
// Host function of Kyber.Encaps.
//
// Copyright (c) 2021 Tatsuki Ono
//
// This software is released under the MIT License.
// https://opensource.org/licenses/mit-license.php
//

#ifndef ATPQC_CUDA_LIB_KYBER_PRIMITIVE_CCAKEM_ENC_CUH_
#define ATPQC_CUDA_LIB_KYBER_PRIMITIVE_CCAKEM_ENC_CUH_

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "../../cuda_debug.hpp"
#include "../../cuda_resource.hpp"
#include "../../rng/wrapper.hpp"
#include "../params.cuh"
#include "common.cuh"
#include "cpapke_enc.cuh"

namespace atpqc_cuda::kyber::primitive::ccakem_enc {

template <class Variant>
struct mem_resource {
  using variant = Variant;
  cuda_resource::pinned_memory<std::uint8_t> rand_host;
  cuda_resource::device_pitched_memory<std::uint8_t> buf;
  cuda_resource::device_pitched_memory<std::uint8_t> kr;
  cpapke_enc::mem_resource<variant> pke_enc_mr;

  mem_resource() = delete;
  mem_resource(unsigned ninputs)
      : rand_host(params::symbytes * ninputs, cudaHostAllocDefault),
        buf(2 * params::symbytes, ninputs),
        kr(2 * params::symbytes, ninputs),
        pke_enc_mr(ninputs) {}
};

template <class Variant, class CpapkeEnc, class Rng, class HashHRand,
          class HashHPk, class HashHCt, class HashG, class Kdf>
class enc {
 private:
  using variant = Variant;
  using rng_type = rng::cuda_hostfunc_wrapper<Rng>;
  using rng_args_type = typename rng_type::args_type;

  unsigned nin;
  CpapkeEnc pke_enc;
  HashHRand hash_rand;
  HashHPk hash_pk;
  HashHCt hash_ct;
  HashG hash_coin;
  Kdf kdf;

 public:
  struct graph_args {
    using pke_enc_args_type = typename CpapkeEnc::graph_args;
    using rng_args_type = typename rng_type::args_type;
    using hash_rand_args_type = typename HashHRand::args_type;
    using hash_pk_args_type = typename HashHPk::args_type;
    using hash_ct_args_type = typename HashHCt::args_type;
    using hash_coin_args_type = typename HashG::args_type;
    using kdf_args_type = typename Kdf::args_type;

    pke_enc_args_type pke_enc_args;

    std::unique_ptr<rng_args_type> randombytes_args;
    std::unique_ptr<hash_rand_args_type> hash_rand_args;
    std::unique_ptr<hash_pk_args_type> hash_pk_args;
    std::unique_ptr<hash_ct_args_type> hash_ct_args;
    std::unique_ptr<hash_coin_args_type> hash_coin_args;
    std::unique_ptr<kdf_args_type> kdf_args;
  };

  graph_args join_graph(cudaGraph_t graph, std::uint8_t* ct,
                        std::size_t ct_pitch, cudaGraphNode_t ct_empty,
                        cudaGraphNode_t* ct_available_ptr, std::uint8_t* ss,
                        std::size_t ss_pitch, cudaGraphNode_t ss_empty,
                        cudaGraphNode_t* ss_available_ptr,
                        const std::uint8_t* pk, std::size_t pk_pitch,
                        cudaGraphNode_t pk_ready, cudaGraphNode_t* pk_used_ptr,
                        const mem_resource<variant>& mr) const {
    graph_args args_pack;
    std::uint8_t* rand_host_ptr = mr.rand_host.get_ptr();
    std::uint8_t* buf_ptr = mr.buf.get_ptr();
    std::size_t buf_pitch = mr.buf.get_pitch();
    std::uint8_t* kr_ptr = mr.kr.get_ptr();
    std::size_t kr_pitch = mr.kr.get_pitch();

    cudaGraphNode_t cpyrand_node;
    {
      cudaMemcpy3DParms memcpy_params;
      memcpy_params.srcPos = make_cudaPos(0, 0, 0);
      memcpy_params.srcPtr = make_cudaPitchedPtr(
          rand_host_ptr, params::symbytes, params::symbytes, nin);
      memcpy_params.srcArray = nullptr;
      memcpy_params.dstPos = make_cudaPos(0, 0, 0);
      memcpy_params.dstPtr =
          make_cudaPitchedPtr(buf_ptr, buf_pitch, 2 * params::symbytes, nin);
      memcpy_params.dstArray = nullptr;
      memcpy_params.extent = make_cudaExtent(params::symbytes, nin, 1);
      memcpy_params.kind = cudaMemcpyHostToDevice;
      CCC(cudaGraphAddMemcpyNode(&cpyrand_node, graph, nullptr, 0,
                                 &memcpy_params));
    }

    cudaGraphNode_t randombytes_node;
    {
      args_pack.randombytes_args = std::make_unique<rng_args_type>(
          rand_host_ptr, params::symbytes * nin);
      cudaHostNodeParams host_params{rng_type::randombytes,
                                     args_pack.randombytes_args.get()};
      std::array dep_array{cpyrand_node};
      CCC(cudaGraphAddHostNode(&randombytes_node, graph, dep_array.data(),
                               dep_array.size(), &host_params));
    }

    cudaGraphNode_t hashrand_node;
    {
      args_pack.hash_rand_args = hash_rand.generate_args(
          buf_ptr, buf_pitch, buf_ptr, buf_pitch, params::symbytes);
      cudaKernelNodeParams kernel_params;
      kernel_params.func = hash_rand.get_func();
      kernel_params.gridDim = hash_rand.get_grid_dim();
      kernel_params.blockDim = hash_rand.get_block_dim();
      kernel_params.sharedMemBytes = hash_rand.get_shared_bytes();
      kernel_params.kernelParams = args_pack.hash_rand_args->get_args_ptr();
      kernel_params.extra = nullptr;
      std::array dep_array{cpyrand_node};
      CCC(cudaGraphAddKernelNode(&hashrand_node, graph, dep_array.data(),
                                 dep_array.size(), &kernel_params));
    }

    cudaGraphNode_t hashpk_node;
    {
      args_pack.hash_pk_args =
          hash_pk.generate_args(buf_ptr + params::symbytes, buf_pitch, pk,
                                pk_pitch, params::publickeybytes<variant>);
      cudaKernelNodeParams kernel_params;
      kernel_params.func = hash_pk.get_func();
      kernel_params.gridDim = hash_pk.get_grid_dim();
      kernel_params.blockDim = hash_pk.get_block_dim();
      kernel_params.sharedMemBytes = hash_pk.get_shared_bytes();
      kernel_params.kernelParams = args_pack.hash_pk_args->get_args_ptr();
      kernel_params.extra = nullptr;
      std::array dep_array{pk_ready};
      CCC(cudaGraphAddKernelNode(&hashpk_node, graph, dep_array.data(),
                                 dep_array.size(), &kernel_params));
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
      std::array dep_array{hashrand_node, hashpk_node};
      CCC(cudaGraphAddKernelNode(&hashcoin_node, graph, dep_array.data(),
                                 dep_array.size(), &kernel_params));
    }

    cudaGraphNode_t pke_enc_c_node;
    cudaGraphNode_t pke_enc_m_node;
    cudaGraphNode_t pke_enc_pk_node;
    cudaGraphNode_t pke_enc_coin_node;
    {
      args_pack.pke_enc_args = pke_enc.join_graph(
          graph, ct, ct_pitch, ct_empty, &pke_enc_c_node, buf_ptr, buf_pitch,
          hashrand_node, &pke_enc_m_node, pk, pk_pitch, pk_ready,
          &pke_enc_pk_node, kr_ptr + params::symbytes, kr_pitch, hashcoin_node,
          &pke_enc_coin_node, mr.pke_enc_mr);
    }

    *ct_available_ptr = pke_enc_c_node;

    {
      std::array dep_array{hashpk_node, pke_enc_pk_node};
      CCC(cudaGraphAddEmptyNode(pk_used_ptr, graph, dep_array.data(),
                                dep_array.size()));
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
      std::array dep_array{pke_enc_c_node, pke_enc_coin_node};
      CCC(cudaGraphAddKernelNode(&hashct_node, graph, dep_array.data(),
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
      std::array dep_array{hashcoin_node, hashct_node, ss_empty};
      CCC(cudaGraphAddKernelNode(&kdf_node, graph, dep_array.data(),
                                 dep_array.size(), &kernel_params));
    }

    *ss_available_ptr = kdf_node;

    return args_pack;
  }

  enc(unsigned ninputs, Variant variant_v, const CpapkeEnc& cpe, Rng rng,
      const HashHRand& hash_h_rand, const HashHPk& hash_h_pk,
      const HashHCt& hash_h_ct, const HashG& hash_g_coin, const Kdf& hash_kdf)
      : nin(ninputs),
        pke_enc(cpe),
        hash_rand(hash_h_rand),
        hash_pk(hash_h_pk),
        hash_ct(hash_h_ct),
        hash_coin(hash_g_coin),
        kdf(hash_kdf) {}
};

}  // namespace atpqc_cuda::kyber::primitive::ccakem_enc

#endif
