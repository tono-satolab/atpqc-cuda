//
// cpapke_dec.cuh
// Host function of Kyber.CPAPKE-Dec.
//
// Copyright (c) 2021 Tatsuki Ono
//
// This software is released under the MIT License.
// https://opensource.org/licenses/mit-license.php
//

#ifndef ATPQC_CUDA_LIB_KYBER_PRIMITIVE_CPAPKE_DEC_CUH_
#define ATPQC_CUDA_LIB_KYBER_PRIMITIVE_CPAPKE_DEC_CUH_

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "../../cuda_debug.hpp"
#include "../../cuda_resource.hpp"
#include "../params.cuh"
#include "common.cuh"

namespace atpqc_cuda::kyber::primitive::cpapke_dec {

template <class Variant>
struct mem_resource {
  using variant = Variant;
  cuda_resource::device_memory<short2> bp;
  cuda_resource::device_memory<short2> skpv;
  cuda_resource::device_memory<short2> v;
  cuda_resource::device_memory<short2> mp;

  mem_resource() = delete;
  mem_resource(unsigned ninputs)
      : bp(poly_size<variant>::vec * ninputs),
        skpv(poly_size<variant>::vec * ninputs),
        v(poly_size<variant>::poly * ninputs),
        mp(poly_size<variant>::poly * ninputs) {}
};

template <class Variant, class FwdNttVec, class InvNttPoly, class VecTimesVec,
          class PolyMinusPoly, class DecodeVec, class DecompressVec,
          class DecompressPoly, class PolyToMsg>
class cpapke_dec {
 private:
  using variant = Variant;

  unsigned nin;
  FwdNttVec ntt_u;
  InvNttPoly intt_su;
  VecTimesVec s_times_u;
  PolyMinusPoly v_minus_su;
  DecodeVec decode_s;
  DecompressVec decompress_u;
  DecompressPoly decompress_v;
  PolyToMsg tomsg;

 public:
  struct graph_args {
    using ntt_u_args_type = typename FwdNttVec::args_type;
    using intt_su_args_type = typename InvNttPoly::args_type;
    using s_times_u_args_type = typename VecTimesVec::args_type;
    using v_minus_su_args_type = typename PolyMinusPoly::args_type;
    using decode_s_args_type = typename DecodeVec::args_type;
    using decompress_u_args_type = typename DecompressVec::args_type;
    using decompress_v_args_type = typename DecompressPoly::args_type;
    using tomsg_args_type = typename PolyToMsg::args_type;

    std::unique_ptr<ntt_u_args_type> ntt_u_args;
    std::unique_ptr<intt_su_args_type> intt_su_args;
    std::unique_ptr<s_times_u_args_type> s_times_u_args;
    std::unique_ptr<v_minus_su_args_type> v_minus_su_args;
    std::unique_ptr<decode_s_args_type> decode_s_args;
    std::unique_ptr<decompress_u_args_type> decompress_u_args;
    std::unique_ptr<decompress_v_args_type> decompress_v_args;
    std::unique_ptr<tomsg_args_type> tomsg_args;
  };

  graph_args join_graph(cudaGraph_t graph, std::uint8_t* m, std::size_t m_pitch,
                        cudaGraphNode_t m_empty,
                        cudaGraphNode_t* m_available_ptr, const std::uint8_t* c,
                        std::size_t c_pitch, cudaGraphNode_t c_ready,
                        cudaGraphNode_t* c_used_ptr, const std::uint8_t* sk,
                        std::size_t sk_pitch, cudaGraphNode_t sk_ready,
                        cudaGraphNode_t* sk_used_ptr,
                        const mem_resource<variant>& mr) const {
    graph_args args_pack;
    short2* bp = mr.bp.get_ptr();
    short2* skpv = mr.skpv.get_ptr();
    short2* v = mr.v.get_ptr();
    short2* mp = mr.mp.get_ptr();

    cudaGraphNode_t decompressu_node;
    {
      args_pack.decompress_u_args = decompress_u.generate_args(bp, c, c_pitch);
      cudaKernelNodeParams kernel_params;
      kernel_params.func = decompress_u.get_func();
      kernel_params.gridDim = decompress_u.get_grid_dim();
      kernel_params.blockDim = decompress_u.get_block_dim();
      kernel_params.sharedMemBytes = decompress_u.get_shared_bytes();
      kernel_params.kernelParams = args_pack.decompress_u_args->get_args_ptr();
      kernel_params.extra = nullptr;
      std::array dep_array{c_ready};
      CCC(cudaGraphAddKernelNode(&decompressu_node, graph, dep_array.data(),
                                 dep_array.size(), &kernel_params));
    }

    cudaGraphNode_t decompressv_node;
    {
      args_pack.decompress_v_args = decompress_v.generate_args(
          v, c + params::polyveccompressedbytes<variant>, c_pitch);
      cudaKernelNodeParams kernel_params;
      kernel_params.func = decompress_v.get_func();
      kernel_params.gridDim = decompress_v.get_grid_dim();
      kernel_params.blockDim = decompress_v.get_block_dim();
      kernel_params.sharedMemBytes = decompress_v.get_shared_bytes();
      kernel_params.kernelParams = args_pack.decompress_v_args->get_args_ptr();
      kernel_params.extra = nullptr;
      std::array dep_array{c_ready};
      CCC(cudaGraphAddKernelNode(&decompressv_node, graph, dep_array.data(),
                                 dep_array.size(), &kernel_params));
    }

    {
      std::array dep_array{decompressu_node, decompressv_node};
      CCC(cudaGraphAddEmptyNode(c_used_ptr, graph, dep_array.data(),
                                dep_array.size()));
    }

    cudaGraphNode_t decodes_node;
    {
      args_pack.decode_s_args = decode_s.generate_args(skpv, sk, sk_pitch);
      cudaKernelNodeParams kernel_params;
      kernel_params.func = decode_s.get_func();
      kernel_params.gridDim = decode_s.get_grid_dim();
      kernel_params.blockDim = decode_s.get_block_dim();
      kernel_params.sharedMemBytes = decode_s.get_shared_bytes();
      kernel_params.kernelParams = args_pack.decode_s_args->get_args_ptr();
      kernel_params.extra = nullptr;
      std::array dep_array{sk_ready};
      CCC(cudaGraphAddKernelNode(&decodes_node, graph, dep_array.data(),
                                 dep_array.size(), &kernel_params));
    }

    *sk_used_ptr = decodes_node;

    cudaGraphNode_t nttu_node;
    {
      args_pack.ntt_u_args = ntt_u.generate_args(bp);
      cudaKernelNodeParams kernel_params;
      kernel_params.func = ntt_u.get_func();
      kernel_params.gridDim = ntt_u.get_grid_dim();
      kernel_params.blockDim = ntt_u.get_block_dim();
      kernel_params.sharedMemBytes = ntt_u.get_shared_bytes();
      kernel_params.kernelParams = args_pack.ntt_u_args->get_args_ptr();
      kernel_params.extra = nullptr;
      std::array dep_array{decompressu_node};
      CCC(cudaGraphAddKernelNode(&nttu_node, graph, dep_array.data(),
                                 dep_array.size(), &kernel_params));
    }

    cudaGraphNode_t stu_node;
    {
      args_pack.s_times_u_args = s_times_u.generate_args(mp, skpv, bp);
      cudaKernelNodeParams kernel_params;
      kernel_params.func = s_times_u.get_func();
      kernel_params.gridDim = s_times_u.get_grid_dim();
      kernel_params.blockDim = s_times_u.get_block_dim();
      kernel_params.sharedMemBytes = s_times_u.get_shared_bytes();
      kernel_params.kernelParams = args_pack.s_times_u_args->get_args_ptr();
      kernel_params.extra = nullptr;
      std::array dep_array{decodes_node, nttu_node};
      CCC(cudaGraphAddKernelNode(&stu_node, graph, dep_array.data(),
                                 dep_array.size(), &kernel_params));
    }

    cudaGraphNode_t inttsu_node;
    {
      args_pack.intt_su_args = intt_su.generate_args(mp);
      cudaKernelNodeParams kernel_params;
      kernel_params.func = intt_su.get_func();
      kernel_params.gridDim = intt_su.get_grid_dim();
      kernel_params.blockDim = intt_su.get_block_dim();
      kernel_params.sharedMemBytes = intt_su.get_shared_bytes();
      kernel_params.kernelParams = args_pack.intt_su_args->get_args_ptr();
      kernel_params.extra = nullptr;
      std::array dep_array{stu_node};
      CCC(cudaGraphAddKernelNode(&inttsu_node, graph, dep_array.data(),
                                 dep_array.size(), &kernel_params));
    }

    cudaGraphNode_t vmsu_node;
    {
      args_pack.v_minus_su_args = v_minus_su.generate_args(mp, v, mp);
      cudaKernelNodeParams kernel_params;
      kernel_params.func = v_minus_su.get_func();
      kernel_params.gridDim = v_minus_su.get_grid_dim();
      kernel_params.blockDim = v_minus_su.get_block_dim();
      kernel_params.sharedMemBytes = v_minus_su.get_shared_bytes();
      kernel_params.kernelParams = args_pack.v_minus_su_args->get_args_ptr();
      kernel_params.extra = nullptr;
      std::array dep_array{inttsu_node, decompressv_node};
      CCC(cudaGraphAddKernelNode(&vmsu_node, graph, dep_array.data(),
                                 dep_array.size(), &kernel_params));
    }

    cudaGraphNode_t tomsg_node;
    {
      args_pack.tomsg_args = tomsg.generate_args(m, m_pitch, mp);
      cudaKernelNodeParams kernel_params;
      kernel_params.func = tomsg.get_func();
      kernel_params.gridDim = tomsg.get_grid_dim();
      kernel_params.blockDim = tomsg.get_block_dim();
      kernel_params.sharedMemBytes = tomsg.get_shared_bytes();
      kernel_params.kernelParams = args_pack.tomsg_args->get_args_ptr();
      kernel_params.extra = nullptr;
      std::array dep_array{vmsu_node, m_empty};
      CCC(cudaGraphAddKernelNode(&tomsg_node, graph, dep_array.data(),
                                 dep_array.size(), &kernel_params));
    }

    *m_available_ptr = tomsg_node;

    return args_pack;
  }

  cpapke_dec(unsigned ninputs, Variant variant_v, const FwdNttVec& fwdntt_vec,
             const InvNttPoly& invntt_poly, const VecTimesVec& vectimesvec,
             const PolyMinusPoly& polyminuspoly, const DecodeVec& decode_vec,
             const DecompressVec& decompress_vec,
             const DecompressPoly& decompress_poly, const PolyToMsg& poly_tomsg)
      : nin(ninputs),
        ntt_u(fwdntt_vec),
        intt_su(invntt_poly),
        s_times_u(vectimesvec),
        v_minus_su(polyminuspoly),
        decode_s(decode_vec),
        decompress_u(decompress_vec),
        decompress_v(decompress_poly),
        tomsg(poly_tomsg) {}
};

}  // namespace atpqc_cuda::kyber::primitive::cpapke_dec

#endif
