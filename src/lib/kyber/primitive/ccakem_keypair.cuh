//
// ccakem_keypair.cuh
// Host function of Kyber.Keypair.
//
// Copyright (c) 2021 Tatsuki Ono
//
// This software is released under the MIT License.
// https://opensource.org/licenses/mit-license.php
//

#ifndef ATPQC_CUDA_LIB_KYBER_PRIMITIVE_CCAKEM_KEYPAIR_CUH_
#define ATPQC_CUDA_LIB_KYBER_PRIMITIVE_CCAKEM_KEYPAIR_CUH_

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "../../cuda_debug.hpp"
#include "../../cuda_resource.hpp"
#include "../../rng/wrapper.hpp"
#include "../params.cuh"
#include "common.cuh"
#include "cpapke_keypair.cuh"

namespace atpqc_cuda::kyber::primitive::ccakem_keypair {

template <class Variant>
struct mem_resource {
  using variant = Variant;
  cuda_resource::pinned_memory<std::uint8_t> rand_host;
  cpapke_keypair::mem_resource<variant> pke_keypair_mr;

  mem_resource() = delete;
  mem_resource(unsigned ninputs)
      : rand_host(params::symbytes * ninputs, cudaHostAllocDefault),
        pke_keypair_mr(ninputs) {}
};

template <class Variant, class CpapkeKeypair, class Rng, class HashH>
class keypair {
 private:
  using variant = Variant;
  using rng_type = rng::cuda_hostfunc_wrapper<Rng>;
  using rng_args_type = typename rng_type::args_type;

  unsigned nin;
  CpapkeKeypair pke_keypair;
  HashH hash_pk;

 public:
  struct graph_args {
    using pke_keypair_args_type = typename CpapkeKeypair::graph_args;
    using rng_args_type = typename rng_type::args_type;
    using hash_pk_args_type = typename HashH::args_type;

    pke_keypair_args_type pke_keypair_args;

    std::unique_ptr<rng_args_type> randombytes_args;
    std::unique_ptr<hash_pk_args_type> hash_pk_args;
  };

  graph_args join_graph(cudaGraph_t graph, std::uint8_t* pk,
                        std::size_t pk_pitch, cudaGraphNode_t pk_empty,
                        cudaGraphNode_t* pk_available_ptr, std::uint8_t* sk,
                        std::size_t sk_pitch, cudaGraphNode_t sk_empty,
                        cudaGraphNode_t* sk_available_ptr,
                        const mem_resource<variant>& mr) const {
    graph_args args_pack;
    std::uint8_t* rand_host_ptr = mr.rand_host.get_ptr();

    cudaGraphNode_t pke_keypair_pk_node;
    cudaGraphNode_t pke_keypair_sk_node;
    {
      args_pack.pke_keypair_args = pke_keypair.join_graph(
          graph, pk, pk_pitch, pk_empty, &pke_keypair_pk_node, sk, sk_pitch,
          sk_empty, &pke_keypair_sk_node, mr.pke_keypair_mr);
    }

    *pk_available_ptr = pke_keypair_pk_node;

    cudaGraphNode_t cpypk_node;
    {
      cudaMemcpy3DParms memcpy_params;
      memcpy_params.srcPos = make_cudaPos(0, 0, 0);
      memcpy_params.srcPtr = make_cudaPitchedPtr(
          pk, pk_pitch, params::publickeybytes<variant>, nin);
      memcpy_params.srcArray = nullptr;
      memcpy_params.dstPos =
          make_cudaPos(params::indcpa_secretkeybytes<variant>, 0, 0);
      memcpy_params.dstPtr = make_cudaPitchedPtr(
          sk, sk_pitch, params::secretkeybytes<variant>, nin);
      memcpy_params.dstArray = nullptr;
      memcpy_params.extent =
          make_cudaExtent(params::indcpa_publickeybytes<variant>, nin, 1);
      memcpy_params.kind = cudaMemcpyDeviceToDevice;
      std::array dep_array{pke_keypair_pk_node, sk_empty};
      CCC(cudaGraphAddMemcpyNode(&cpypk_node, graph, dep_array.data(),
                                 dep_array.size(), &memcpy_params));
    }

    cudaGraphNode_t hashpk_node;
    {
      args_pack.hash_pk_args = hash_pk.generate_args(
          sk + (params::secretkeybytes<variant> - 2 * params::symbytes),
          sk_pitch, pk, pk_pitch, params::publickeybytes<variant>);
      cudaKernelNodeParams kernel_params;
      kernel_params.func = hash_pk.get_func();
      kernel_params.gridDim = hash_pk.get_grid_dim();
      kernel_params.blockDim = hash_pk.get_block_dim();
      kernel_params.sharedMemBytes = hash_pk.get_shared_bytes();
      kernel_params.kernelParams = args_pack.hash_pk_args->get_args_ptr();
      kernel_params.extra = nullptr;
      std::array dep_array{pke_keypair_pk_node, sk_empty};
      CCC(cudaGraphAddKernelNode(&hashpk_node, graph, dep_array.data(),
                                 dep_array.size(), &kernel_params));
    }

    cudaGraphNode_t cpyrand_node;
    {
      cudaMemcpy3DParms memcpy_params;
      memcpy_params.srcPos = make_cudaPos(0, 0, 0);
      memcpy_params.srcPtr = make_cudaPitchedPtr(
          rand_host_ptr, params::symbytes, params::symbytes, nin);
      memcpy_params.srcArray = nullptr;
      memcpy_params.dstPos = make_cudaPos(
          params::secretkeybytes<variant> - params::symbytes, 0, 0);
      memcpy_params.dstPtr = make_cudaPitchedPtr(
          sk, sk_pitch, params::secretkeybytes<variant>, nin);
      memcpy_params.dstArray = nullptr;
      memcpy_params.extent = make_cudaExtent(params::symbytes, nin, 1);
      memcpy_params.kind = cudaMemcpyHostToDevice;
      std::array dep_array{sk_empty};
      CCC(cudaGraphAddMemcpyNode(&cpyrand_node, graph, dep_array.data(),
                                 dep_array.size(), &memcpy_params));
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

    {
      std::array dep_array{pke_keypair_sk_node, cpypk_node, hashpk_node,
                           cpyrand_node};
      CCC(cudaGraphAddEmptyNode(sk_available_ptr, graph, dep_array.data(),
                                dep_array.size()));
    }

    return args_pack;
  }

  keypair(unsigned ninputs, Variant variant_v, const CpapkeKeypair& cpk,
          Rng rng, const HashH& hash_h)
      : nin(ninputs), pke_keypair(cpk), hash_pk(hash_h) {}
};

}  // namespace atpqc_cuda::kyber::primitive::ccakem_keypair

#endif
