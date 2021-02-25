//
// cpapke_keypair.cuh
// Host function of Kyber.CPAPKE-Keypair.
//
// Copyright (c) 2021 Tatsuki Ono
//
// This software is released under the MIT License.
// https://opensource.org/licenses/mit-license.php
//

#ifndef ATPQC_CUDA_LIB_KYBER_PRIMITIVE_CPAPKE_KEYPAIR_CUH_
#define ATPQC_CUDA_LIB_KYBER_PRIMITIVE_CPAPKE_KEYPAIR_CUH_

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "../../cuda_debug.hpp"
#include "../../cuda_resource.hpp"
#include "../../rng/wrapper.hpp"
#include "../params.cuh"
#include "common.cuh"

namespace atpqc_cuda::kyber::primitive::cpapke_keypair {

template <class Variant>
struct mem_resource {
  using variant = Variant;
  cuda_resource::pinned_memory<std::uint8_t> rand_host;
  cuda_resource::device_pitched_memory<std::uint8_t> seed;
  cuda_resource::device_memory<short2> a;
  cuda_resource::device_memory<short2> e;
  cuda_resource::device_memory<short2> pkpv;
  cuda_resource::device_memory<short2> skpv;

  mem_resource() = delete;
  mem_resource(unsigned ninputs)
      : rand_host(params::symbytes * ninputs, cudaHostAllocDefault),
        seed(2 * params::symbytes, ninputs),
        a(poly_size<variant>::mat * ninputs),
        e(poly_size<variant>::vec * ninputs),
        pkpv(poly_size<variant>::vec * ninputs),
        skpv(poly_size<variant>::vec * ninputs) {}
};

template <class Variant, class Rng, class HashG, class GenMat, class GenVecS,
          class GenVecE, class FwdNttVecS, class FwdNttVecE,
          class MatTimesVecToMontPlusVec, class EncodeVecT, class EncodeVecS>
class cpapke_keypair {
 private:
  using variant = Variant;
  using rng_type = rng::cuda_hostfunc_wrapper<Rng>;
  using rng_args_type = typename rng_type::args_type;

  static constexpr unsigned pseed_offset = 0;
  static constexpr unsigned nseed_offset = params::symbytes;

  unsigned nin;
  HashG hash_seed;
  GenMat generate_a;
  GenVecS generate_s;
  GenVecE generate_e;
  FwdNttVecS ntt_s;
  FwdNttVecE ntt_e;
  MatTimesVecToMontPlusVec a_times_s_plus_e;
  EncodeVecT encode_t;
  EncodeVecS encode_s;

 public:
  struct graph_args {
    using rng_args_type = typename rng_type::args_type;
    using hash_seed_args_type = typename HashG::args_type;
    using generate_a_args_type = typename GenMat::args_type;
    using generate_s_args_type = typename GenVecS::args_type;
    using generate_e_args_type = typename GenVecE::args_type;
    using ntt_s_args_type = typename FwdNttVecS::args_type;
    using ntt_e_args_type = typename FwdNttVecE::args_type;
    using atspe_args_type = typename MatTimesVecToMontPlusVec::args_type;
    using encode_t_args_type = typename EncodeVecT::args_type;
    using encode_s_args_type = typename EncodeVecS::args_type;

    std::unique_ptr<rng_args_type> randombytes_args;
    std::unique_ptr<hash_seed_args_type> hash_seed_args;
    std::unique_ptr<generate_a_args_type> generate_a_args;
    std::unique_ptr<generate_s_args_type> generate_s_args;
    std::unique_ptr<generate_e_args_type> generate_e_args;
    std::unique_ptr<ntt_s_args_type> ntt_s_args;
    std::unique_ptr<ntt_e_args_type> ntt_e_args;
    std::unique_ptr<atspe_args_type> atspe_args;
    std::unique_ptr<encode_t_args_type> encode_t_args;
    std::unique_ptr<encode_s_args_type> encode_s_args;
  };

  graph_args join_graph(cudaGraph_t graph, std::uint8_t* pk,
                        std::size_t pk_pitch, cudaGraphNode_t pk_empty,
                        cudaGraphNode_t* pk_available_ptr, std::uint8_t* sk,
                        std::size_t sk_pitch, cudaGraphNode_t sk_empty,
                        cudaGraphNode_t* sk_available_ptr,
                        const mem_resource<variant>& mr) const {
    graph_args args_pack;
    std::uint8_t* rand_host_ptr = mr.rand_host.get_ptr();
    std::uint8_t* seed_ptr = mr.seed.get_ptr();
    std::size_t seed_pitch = mr.seed.get_pitch();
    std::uint8_t* pseed = seed_ptr + pseed_offset;
    std::uint8_t* nseed = seed_ptr + nseed_offset;
    short2* a = mr.a.get_ptr();
    short2* e = mr.e.get_ptr();
    short2* pkpv = mr.pkpv.get_ptr();
    short2* skpv = mr.skpv.get_ptr();

    cudaGraphNode_t cpyrand_node;
    {
      cudaMemcpy3DParms memcpy_params;
      memcpy_params.srcPos = make_cudaPos(0, 0, 0);
      memcpy_params.srcPtr = make_cudaPitchedPtr(
          rand_host_ptr, params::symbytes, params::symbytes, nin);
      memcpy_params.srcArray = nullptr;
      memcpy_params.dstPos = make_cudaPos(0, 0, 0);
      memcpy_params.dstPtr =
          make_cudaPitchedPtr(seed_ptr, seed_pitch, 2 * params::symbytes, nin);
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

    cudaGraphNode_t hashseed_node;
    {
      args_pack.hash_seed_args = hash_seed.generate_args(
          seed_ptr, seed_pitch, seed_ptr, seed_pitch, params::symbytes);
      cudaKernelNodeParams kernel_params;
      kernel_params.func = hash_seed.get_func();
      kernel_params.gridDim = hash_seed.get_grid_dim();
      kernel_params.blockDim = hash_seed.get_block_dim();
      kernel_params.sharedMemBytes = hash_seed.get_shared_bytes();
      kernel_params.kernelParams = args_pack.hash_seed_args->get_args_ptr();
      kernel_params.extra = nullptr;
      std::array dep_array{cpyrand_node};
      CCC(cudaGraphAddKernelNode(&hashseed_node, graph, dep_array.data(),
                                 dep_array.size(), &kernel_params));
    }

    cudaGraphNode_t gena_node;
    {
      args_pack.generate_a_args =
          generate_a.generate_args(a, pseed, seed_pitch);
      cudaKernelNodeParams kernel_params;
      kernel_params.func = generate_a.get_func();
      kernel_params.gridDim = generate_a.get_grid_dim();
      kernel_params.blockDim = generate_a.get_block_dim();
      kernel_params.sharedMemBytes = generate_a.get_shared_bytes();
      kernel_params.kernelParams = args_pack.generate_a_args->get_args_ptr();
      kernel_params.extra = nullptr;
      std::array dep_array{hashseed_node};
      CCC(cudaGraphAddKernelNode(&gena_node, graph, dep_array.data(),
                                 dep_array.size(), &kernel_params));
    }

    cudaGraphNode_t gens_node;
    {
      args_pack.generate_s_args =
          generate_s.generate_args(skpv, nseed, seed_pitch, 0);
      cudaKernelNodeParams kernel_params;
      kernel_params.func = generate_s.get_func();
      kernel_params.gridDim = generate_s.get_grid_dim();
      kernel_params.blockDim = generate_s.get_block_dim();
      kernel_params.sharedMemBytes = generate_s.get_shared_bytes();
      kernel_params.kernelParams = args_pack.generate_s_args->get_args_ptr();
      kernel_params.extra = nullptr;
      std::array dep_array{hashseed_node};
      CCC(cudaGraphAddKernelNode(&gens_node, graph, dep_array.data(),
                                 dep_array.size(), &kernel_params));
    }

    cudaGraphNode_t gene_node;
    {
      args_pack.generate_e_args =
          generate_e.generate_args(e, nseed, seed_pitch, params::k<variant>);
      cudaKernelNodeParams kernel_params;
      kernel_params.func = generate_e.get_func();
      kernel_params.gridDim = generate_e.get_grid_dim();
      kernel_params.blockDim = generate_e.get_block_dim();
      kernel_params.sharedMemBytes = generate_e.get_shared_bytes();
      kernel_params.kernelParams = args_pack.generate_e_args->get_args_ptr();
      kernel_params.extra = nullptr;
      std::array dep_array{hashseed_node};
      CCC(cudaGraphAddKernelNode(&gene_node, graph, dep_array.data(),
                                 dep_array.size(), &kernel_params));
    }

    cudaGraphNode_t ntts_node;
    {
      args_pack.ntt_s_args = ntt_s.generate_args(skpv);
      cudaKernelNodeParams kernel_params;
      kernel_params.func = ntt_s.get_func();
      kernel_params.gridDim = ntt_s.get_grid_dim();
      kernel_params.blockDim = ntt_s.get_block_dim();
      kernel_params.sharedMemBytes = ntt_s.get_shared_bytes();
      kernel_params.kernelParams = args_pack.ntt_s_args->get_args_ptr();
      kernel_params.extra = nullptr;
      std::array dep_array{gens_node};
      CCC(cudaGraphAddKernelNode(&ntts_node, graph, dep_array.data(),
                                 dep_array.size(), &kernel_params));
    }

    cudaGraphNode_t ntte_node;
    {
      args_pack.ntt_e_args = ntt_e.generate_args(e);
      cudaKernelNodeParams kernel_params;
      kernel_params.func = ntt_e.get_func();
      kernel_params.gridDim = ntt_e.get_grid_dim();
      kernel_params.blockDim = ntt_e.get_block_dim();
      kernel_params.sharedMemBytes = ntt_e.get_shared_bytes();
      kernel_params.kernelParams = args_pack.ntt_e_args->get_args_ptr();
      kernel_params.extra = nullptr;
      std::array dep_array{gene_node};
      CCC(cudaGraphAddKernelNode(&ntte_node, graph, dep_array.data(),
                                 dep_array.size(), &kernel_params));
    }

    cudaGraphNode_t atspe_node;
    {
      args_pack.atspe_args = a_times_s_plus_e.generate_args(pkpv, a, skpv, e);
      cudaKernelNodeParams kernel_params;
      kernel_params.func = a_times_s_plus_e.get_func();
      kernel_params.gridDim = a_times_s_plus_e.get_grid_dim();
      kernel_params.blockDim = a_times_s_plus_e.get_block_dim();
      kernel_params.sharedMemBytes = a_times_s_plus_e.get_shared_bytes();
      kernel_params.kernelParams = args_pack.atspe_args->get_args_ptr();
      kernel_params.extra = nullptr;
      std::array dep_array{gena_node, ntts_node, ntte_node};
      CCC(cudaGraphAddKernelNode(&atspe_node, graph, dep_array.data(),
                                 dep_array.size(), &kernel_params));
    }

    cudaGraphNode_t encodes_node;
    {
      args_pack.encode_s_args = encode_s.generate_args(sk, sk_pitch, skpv);
      cudaKernelNodeParams kernel_params;
      kernel_params.func = encode_s.get_func();
      kernel_params.gridDim = encode_s.get_grid_dim();
      kernel_params.blockDim = encode_s.get_block_dim();
      kernel_params.sharedMemBytes = encode_s.get_shared_bytes();
      kernel_params.kernelParams = args_pack.encode_s_args->get_args_ptr();
      kernel_params.extra = nullptr;
      std::array dep_array{ntts_node, sk_empty};
      CCC(cudaGraphAddKernelNode(&encodes_node, graph, dep_array.data(),
                                 dep_array.size(), &kernel_params));
    }

    *sk_available_ptr = encodes_node;

    cudaGraphNode_t encodet_node;
    {
      args_pack.encode_t_args = encode_t.generate_args(pk, pk_pitch, pkpv);
      cudaKernelNodeParams kernel_params;
      kernel_params.func = encode_t.get_func();
      kernel_params.gridDim = encode_t.get_grid_dim();
      kernel_params.blockDim = encode_t.get_block_dim();
      kernel_params.sharedMemBytes = encode_t.get_shared_bytes();
      kernel_params.kernelParams = args_pack.encode_t_args->get_args_ptr();
      kernel_params.extra = nullptr;
      std::array dep_array{atspe_node, pk_empty};
      CCC(cudaGraphAddKernelNode(&encodet_node, graph, dep_array.data(),
                                 dep_array.size(), &kernel_params));
    }

    cudaGraphNode_t pseedcpy_node;
    {
      cudaMemcpy3DParms memcpy_params;
      memcpy_params.srcPos = make_cudaPos(0, 0, 0);
      memcpy_params.srcPtr =
          make_cudaPitchedPtr(pseed, seed_pitch, 2 * params::symbytes, nin);
      memcpy_params.srcArray = nullptr;
      memcpy_params.dstPos = make_cudaPos(params::polyvecbytes<variant>, 0, 0);
      memcpy_params.dstPtr = make_cudaPitchedPtr(
          pk, pk_pitch, params::indcpa_publickeybytes<variant>, nin);
      memcpy_params.dstArray = nullptr;
      memcpy_params.extent = make_cudaExtent(params::symbytes, nin, 1);
      memcpy_params.kind = cudaMemcpyDeviceToDevice;
      std::array dep_array{hashseed_node, pk_empty};
      CCC(cudaGraphAddMemcpyNode(&pseedcpy_node, graph, dep_array.data(),
                                 dep_array.size(), &memcpy_params));
    }

    {
      std::array dep_array{encodet_node, pseedcpy_node};
      CCC(cudaGraphAddEmptyNode(pk_available_ptr, graph, dep_array.data(),
                                dep_array.size()));
    }

    return args_pack;
  }

  cpapke_keypair(unsigned ninputs, Variant variant_v, Rng rng,
                 const HashG& hash_g, const GenMat& gen_mat,
                 const GenVecS& gen_vec_s, const GenVecE& gen_vec_e,
                 const FwdNttVecS& fwdntt_vec_s, const FwdNttVecE& fwdntt_vec_e,
                 const MatTimesVecToMontPlusVec& mattimesvec_tomont_plusvec,
                 const EncodeVecT& encode_vec_t, const EncodeVecS& encode_vec_s)
      : nin(ninputs),
        hash_seed(hash_g),
        generate_a(gen_mat),
        generate_s(gen_vec_s),
        generate_e(gen_vec_e),
        ntt_s(fwdntt_vec_s),
        ntt_e(fwdntt_vec_e),
        a_times_s_plus_e(mattimesvec_tomont_plusvec),
        encode_t(encode_vec_t),
        encode_s(encode_vec_s) {}
};

}  // namespace atpqc_cuda::kyber::primitive::cpapke_keypair

#endif
