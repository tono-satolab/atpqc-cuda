//
// cpapke_enc.cuh
// Host function of Kyber.CPAPKE-Enc.
//
// Copyright (c) 2021 Tatsuki Ono
//
// This software is released under the MIT License.
// https://opensource.org/licenses/mit-license.php
//

#ifndef ATPQC_CUDA_LIB_KYBER_PRIMITIVE_CPAPKE_ENC_CUH_
#define ATPQC_CUDA_LIB_KYBER_PRIMITIVE_CPAPKE_ENC_CUH_

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "../../cuda_debug.hpp"
#include "../../cuda_resource.hpp"
#include "../params.cuh"
#include "common.cuh"

namespace atpqc_cuda::kyber::primitive::cpapke_enc {

template <class Variant>
struct mem_resource {
  using variant = Variant;
  cuda_resource::device_memory<short2> at;
  cuda_resource::device_memory<short2> sp;
  cuda_resource::device_memory<short2> pkpv;
  cuda_resource::device_memory<short2> ep;
  cuda_resource::device_memory<short2> bp;
  cuda_resource::device_memory<short2> v;
  cuda_resource::device_memory<short2> k;
  cuda_resource::device_memory<short2> epp;

  mem_resource() = delete;
  mem_resource(unsigned ninputs)
      : at(poly_size<variant>::mat * ninputs),
        sp(poly_size<variant>::vec * ninputs),
        pkpv(poly_size<variant>::vec * ninputs),
        ep(poly_size<variant>::vec * ninputs),
        bp(poly_size<variant>::vec * ninputs),
        v(poly_size<variant>::poly * ninputs),
        k(poly_size<variant>::poly * ninputs),
        epp(poly_size<variant>::poly * ninputs) {}
};

template <class Variant, class GenMatTransposed, class GenVecR, class GenVecE1,
          class GenPoly, class FwdNttVecR, class InvNttVecARToMont,
          class InvNttPolyTRToMont, class MatTimesVec, class VecTimesVec,
          class VecPlusVec, class PolyPlusPolyPlusPoly, class DecodeVec,
          class PolyFromMsg, class CompressVec, class CompressPoly>
class cpapke_enc {
 private:
  using variant = Variant;

  unsigned nin;
  GenMatTransposed generate_at;
  GenVecR generate_r;
  GenVecE1 generate_e1;
  GenPoly generate_e2;
  FwdNttVecR ntt_r;
  InvNttVecARToMont intt_ar;
  InvNttPolyTRToMont intt_tr;
  MatTimesVec a_times_r;
  VecTimesVec t_times_r;
  VecPlusVec ar_plus_e;
  PolyPlusPolyPlusPoly tr_plus_e2_plus_m;
  DecodeVec decode_t;
  PolyFromMsg frommsg;
  CompressVec compress_u;
  CompressPoly compress_v;

 public:
  struct graph_args {
    using generate_at_args_type = typename GenMatTransposed::args_type;
    using generate_r_args_type = typename GenVecR::args_type;
    using generate_e1_args_type = typename GenVecE1::args_type;
    using generate_e2_args_type = typename GenPoly::args_type;
    using ntt_r_args_type = typename FwdNttVecR::args_type;
    using intt_ar_args_type = typename InvNttVecARToMont::args_type;
    using intt_tr_args_type = typename InvNttPolyTRToMont::args_type;
    using a_times_r_args_type = typename MatTimesVec::args_type;
    using t_times_r_args_type = typename VecTimesVec::args_type;
    using ar_plus_e_args_type = typename VecPlusVec::args_type;
    using tr_plus_e2_plus_m_args_type =
        typename PolyPlusPolyPlusPoly::args_type;
    using decode_t_args_type = typename DecodeVec::args_type;
    using frommsg_args_type = typename PolyFromMsg::args_type;
    using compress_u_args_type = typename CompressVec::args_type;
    using compress_v_args_type = typename CompressPoly::args_type;

    std::unique_ptr<generate_at_args_type> generate_at_args;
    std::unique_ptr<generate_r_args_type> generate_r_args;
    std::unique_ptr<generate_e1_args_type> generate_e1_args;
    std::unique_ptr<generate_e2_args_type> generate_e2_args;
    std::unique_ptr<ntt_r_args_type> ntt_r_args;
    std::unique_ptr<intt_ar_args_type> intt_ar_args;
    std::unique_ptr<intt_tr_args_type> intt_tr_args;
    std::unique_ptr<a_times_r_args_type> a_times_r_args;
    std::unique_ptr<t_times_r_args_type> t_times_r_args;
    std::unique_ptr<ar_plus_e_args_type> ar_plus_e_args;
    std::unique_ptr<tr_plus_e2_plus_m_args_type> tr_plus_e2_plus_m_args;
    std::unique_ptr<decode_t_args_type> decode_t_args;
    std::unique_ptr<frommsg_args_type> frommsg_args;
    std::unique_ptr<compress_u_args_type> compress_u_args;
    std::unique_ptr<compress_v_args_type> compress_v_args;
  };

  graph_args join_graph(cudaGraph_t graph, std::uint8_t* c, std::size_t c_pitch,
                        cudaGraphNode_t c_empty,
                        cudaGraphNode_t* c_available_ptr, const std::uint8_t* m,
                        std::size_t m_pitch, cudaGraphNode_t m_ready,
                        cudaGraphNode_t* m_used_ptr, const std::uint8_t* pk,
                        std::size_t pk_pitch, cudaGraphNode_t pk_ready,
                        cudaGraphNode_t* pk_used_ptr, const std::uint8_t* coins,
                        std::size_t coins_pitch, cudaGraphNode_t coins_ready,
                        cudaGraphNode_t* coins_used_ptr,
                        const mem_resource<variant>& mr) const {
    graph_args args_pack;
    short2* at = mr.at.get_ptr();
    short2* sp = mr.sp.get_ptr();
    short2* pkpv = mr.pkpv.get_ptr();
    short2* ep = mr.ep.get_ptr();
    short2* bp = mr.bp.get_ptr();
    short2* v = mr.v.get_ptr();
    short2* k = mr.k.get_ptr();
    short2* epp = mr.epp.get_ptr();

    cudaGraphNode_t decodet_node;
    {
      args_pack.decode_t_args = decode_t.generate_args(pkpv, pk, pk_pitch);
      cudaKernelNodeParams kernel_params;
      kernel_params.func = decode_t.get_func();
      kernel_params.gridDim = decode_t.get_grid_dim();
      kernel_params.blockDim = decode_t.get_block_dim();
      kernel_params.sharedMemBytes = decode_t.get_shared_bytes();
      kernel_params.kernelParams = args_pack.decode_t_args->get_args_ptr();
      kernel_params.extra = nullptr;
      std::array dep_array{pk_ready};
      CCC(cudaGraphAddKernelNode(&decodet_node, graph, dep_array.data(),
                                 dep_array.size(), &kernel_params));
    }

    cudaGraphNode_t frommsg_node;
    {
      args_pack.frommsg_args = frommsg.generate_args(k, m, m_pitch);
      cudaKernelNodeParams kernel_params;
      kernel_params.func = frommsg.get_func();
      kernel_params.gridDim = frommsg.get_grid_dim();
      kernel_params.blockDim = frommsg.get_block_dim();
      kernel_params.sharedMemBytes = frommsg.get_shared_bytes();
      kernel_params.kernelParams = args_pack.frommsg_args->get_args_ptr();
      kernel_params.extra = nullptr;
      std::array dep_array{m_ready};
      CCC(cudaGraphAddKernelNode(&frommsg_node, graph, dep_array.data(),
                                 dep_array.size(), &kernel_params));
    }

    *m_used_ptr = frommsg_node;

    cudaGraphNode_t generateat_node;
    {
      args_pack.generate_at_args = generate_at.generate_args(
          at, pk + params::polyvecbytes<variant>, pk_pitch);
      cudaKernelNodeParams kernel_params;
      kernel_params.func = generate_at.get_func();
      kernel_params.gridDim = generate_at.get_grid_dim();
      kernel_params.blockDim = generate_at.get_block_dim();
      kernel_params.sharedMemBytes = generate_at.get_shared_bytes();
      kernel_params.kernelParams = args_pack.generate_at_args->get_args_ptr();
      kernel_params.extra = nullptr;
      std::array dep_array{pk_ready};
      CCC(cudaGraphAddKernelNode(&generateat_node, graph, dep_array.data(),
                                 dep_array.size(), &kernel_params));
    }

    {
      std::array dep_array{decodet_node, generateat_node};
      CCC(cudaGraphAddEmptyNode(pk_used_ptr, graph, dep_array.data(),
                                dep_array.size()));
    }

    cudaGraphNode_t generater_node;
    {
      args_pack.generate_r_args =
          generate_r.generate_args(sp, coins, coins_pitch, 0);
      cudaKernelNodeParams kernel_params;
      kernel_params.func = generate_r.get_func();
      kernel_params.gridDim = generate_r.get_grid_dim();
      kernel_params.blockDim = generate_r.get_block_dim();
      kernel_params.sharedMemBytes = generate_r.get_shared_bytes();
      kernel_params.kernelParams = args_pack.generate_r_args->get_args_ptr();
      kernel_params.extra = nullptr;
      std::array dep_array{coins_ready};
      CCC(cudaGraphAddKernelNode(&generater_node, graph, dep_array.data(),
                                 dep_array.size(), &kernel_params));
    }

    cudaGraphNode_t generatee1_node;
    {
      args_pack.generate_e1_args =
          generate_e1.generate_args(ep, coins, coins_pitch, params::k<variant>);
      cudaKernelNodeParams kernel_params;
      kernel_params.func = generate_e1.get_func();
      kernel_params.gridDim = generate_e1.get_grid_dim();
      kernel_params.blockDim = generate_e1.get_block_dim();
      kernel_params.sharedMemBytes = generate_e1.get_shared_bytes();
      kernel_params.kernelParams = args_pack.generate_e1_args->get_args_ptr();
      kernel_params.extra = nullptr;
      std::array dep_array{coins_ready};
      CCC(cudaGraphAddKernelNode(&generatee1_node, graph, dep_array.data(),
                                 dep_array.size(), &kernel_params));
    }

    cudaGraphNode_t generatee2_node;
    {
      args_pack.generate_e2_args = generate_e2.generate_args(
          epp, coins, coins_pitch, params::k<variant> * 2);
      cudaKernelNodeParams kernel_params;
      kernel_params.func = generate_e2.get_func();
      kernel_params.gridDim = generate_e2.get_grid_dim();
      kernel_params.blockDim = generate_e2.get_block_dim();
      kernel_params.sharedMemBytes = generate_e2.get_shared_bytes();
      kernel_params.kernelParams = args_pack.generate_e2_args->get_args_ptr();
      kernel_params.extra = nullptr;
      std::array dep_array{coins_ready};
      CCC(cudaGraphAddKernelNode(&generatee2_node, graph, dep_array.data(),
                                 dep_array.size(), &kernel_params));
    }

    {
      std::array dep_array{generater_node, generatee1_node, generatee2_node};
      CCC(cudaGraphAddEmptyNode(coins_used_ptr, graph, dep_array.data(),
                                dep_array.size()));
    }

    cudaGraphNode_t nttr_node;
    {
      args_pack.ntt_r_args = ntt_r.generate_args(sp);
      cudaKernelNodeParams kernel_params;
      kernel_params.func = ntt_r.get_func();
      kernel_params.gridDim = ntt_r.get_grid_dim();
      kernel_params.blockDim = ntt_r.get_block_dim();
      kernel_params.sharedMemBytes = ntt_r.get_shared_bytes();
      kernel_params.kernelParams = args_pack.ntt_r_args->get_args_ptr();
      kernel_params.extra = nullptr;
      std::array dep_array{generater_node};
      CCC(cudaGraphAddKernelNode(&nttr_node, graph, dep_array.data(),
                                 dep_array.size(), &kernel_params));
    }

    cudaGraphNode_t atr_node;
    {
      args_pack.a_times_r_args = a_times_r.generate_args(bp, at, sp);
      cudaKernelNodeParams kernel_params;
      kernel_params.func = a_times_r.get_func();
      kernel_params.gridDim = a_times_r.get_grid_dim();
      kernel_params.blockDim = a_times_r.get_block_dim();
      kernel_params.sharedMemBytes = a_times_r.get_shared_bytes();
      kernel_params.kernelParams = args_pack.a_times_r_args->get_args_ptr();
      kernel_params.extra = nullptr;
      std::array dep_array{generateat_node, nttr_node};
      CCC(cudaGraphAddKernelNode(&atr_node, graph, dep_array.data(),
                                 dep_array.size(), &kernel_params));
    }

    cudaGraphNode_t ttr_node;
    {
      args_pack.t_times_r_args = t_times_r.generate_args(v, pkpv, sp);
      cudaKernelNodeParams kernel_params;
      kernel_params.func = t_times_r.get_func();
      kernel_params.gridDim = t_times_r.get_grid_dim();
      kernel_params.blockDim = t_times_r.get_block_dim();
      kernel_params.sharedMemBytes = t_times_r.get_shared_bytes();
      kernel_params.kernelParams = args_pack.t_times_r_args->get_args_ptr();
      kernel_params.extra = nullptr;
      std::array dep_array{decodet_node, nttr_node};
      CCC(cudaGraphAddKernelNode(&ttr_node, graph, dep_array.data(),
                                 dep_array.size(), &kernel_params));
    }

    cudaGraphNode_t inttar_node;
    {
      args_pack.intt_ar_args = intt_ar.generate_args(bp);
      cudaKernelNodeParams kernel_params;
      kernel_params.func = intt_ar.get_func();
      kernel_params.gridDim = intt_ar.get_grid_dim();
      kernel_params.blockDim = intt_ar.get_block_dim();
      kernel_params.sharedMemBytes = intt_ar.get_shared_bytes();
      kernel_params.kernelParams = args_pack.intt_ar_args->get_args_ptr();
      kernel_params.extra = nullptr;
      std::array dep_array{atr_node};
      CCC(cudaGraphAddKernelNode(&inttar_node, graph, dep_array.data(),
                                 dep_array.size(), &kernel_params));
    }

    cudaGraphNode_t intttr_node;
    {
      args_pack.intt_tr_args = intt_tr.generate_args(v);
      cudaKernelNodeParams kernel_params;
      kernel_params.func = intt_tr.get_func();
      kernel_params.gridDim = intt_tr.get_grid_dim();
      kernel_params.blockDim = intt_tr.get_block_dim();
      kernel_params.sharedMemBytes = intt_tr.get_shared_bytes();
      kernel_params.kernelParams = args_pack.intt_tr_args->get_args_ptr();
      kernel_params.extra = nullptr;
      std::array dep_array{ttr_node};
      CCC(cudaGraphAddKernelNode(&intttr_node, graph, dep_array.data(),
                                 dep_array.size(), &kernel_params));
    }

    cudaGraphNode_t arpe_node;
    {
      args_pack.ar_plus_e_args = ar_plus_e.generate_args(bp, bp, ep);
      cudaKernelNodeParams kernel_params;
      kernel_params.func = ar_plus_e.get_func();
      kernel_params.gridDim = ar_plus_e.get_grid_dim();
      kernel_params.blockDim = ar_plus_e.get_block_dim();
      kernel_params.sharedMemBytes = ar_plus_e.get_shared_bytes();
      kernel_params.kernelParams = args_pack.ar_plus_e_args->get_args_ptr();
      kernel_params.extra = nullptr;
      std::array dep_array{inttar_node, generatee1_node};
      CCC(cudaGraphAddKernelNode(&arpe_node, graph, dep_array.data(),
                                 dep_array.size(), &kernel_params));
    }

    cudaGraphNode_t trpe2pm_node;
    {
      args_pack.tr_plus_e2_plus_m_args =
          tr_plus_e2_plus_m.generate_args(v, v, epp, k);
      cudaKernelNodeParams kernel_params;
      kernel_params.func = tr_plus_e2_plus_m.get_func();
      kernel_params.gridDim = tr_plus_e2_plus_m.get_grid_dim();
      kernel_params.blockDim = tr_plus_e2_plus_m.get_block_dim();
      kernel_params.sharedMemBytes = tr_plus_e2_plus_m.get_shared_bytes();
      kernel_params.kernelParams =
          args_pack.tr_plus_e2_plus_m_args->get_args_ptr();
      kernel_params.extra = nullptr;
      std::array dep_array{intttr_node, generatee2_node, frommsg_node};
      CCC(cudaGraphAddKernelNode(&trpe2pm_node, graph, dep_array.data(),
                                 dep_array.size(), &kernel_params));
    }

    cudaGraphNode_t compressu_node;
    {
      args_pack.compress_u_args = compress_u.generate_args(c, c_pitch, bp);
      cudaKernelNodeParams kernel_params;
      kernel_params.func = compress_u.get_func();
      kernel_params.gridDim = compress_u.get_grid_dim();
      kernel_params.blockDim = compress_u.get_block_dim();
      kernel_params.sharedMemBytes = compress_u.get_shared_bytes();
      kernel_params.kernelParams = args_pack.compress_u_args->get_args_ptr();
      kernel_params.extra = nullptr;
      std::array dep_array{arpe_node, c_empty};
      CCC(cudaGraphAddKernelNode(&compressu_node, graph, dep_array.data(),
                                 dep_array.size(), &kernel_params));
    }

    cudaGraphNode_t compressv_node;
    {
      args_pack.compress_v_args = compress_v.generate_args(
          c + params::polyveccompressedbytes<variant>, c_pitch, v);
      cudaKernelNodeParams kernel_params;
      kernel_params.func = compress_v.get_func();
      kernel_params.gridDim = compress_v.get_grid_dim();
      kernel_params.blockDim = compress_v.get_block_dim();
      kernel_params.sharedMemBytes = compress_v.get_shared_bytes();
      kernel_params.kernelParams = args_pack.compress_v_args->get_args_ptr();
      kernel_params.extra = nullptr;
      std::array dep_array{trpe2pm_node, c_empty};
      CCC(cudaGraphAddKernelNode(&compressv_node, graph, dep_array.data(),
                                 dep_array.size(), &kernel_params));
    }

    {
      std::array dep_array{compressu_node, compressv_node};
      CCC(cudaGraphAddEmptyNode(c_available_ptr, graph, dep_array.data(),
                                dep_array.size()));
    }

    return args_pack;
  }

  cpapke_enc(unsigned ninputs, Variant variant_v,
             const GenMatTransposed& gen_mat_transposed,
             const GenVecR& gen_vec_r, const GenVecE1& gen_vec_e1,
             const GenPoly& gen_poly, const FwdNttVecR& fwdntt_vec_r,
             const InvNttVecARToMont& invntt_vec_ar_tomont,
             const InvNttPolyTRToMont& invntt_poly_tr_tomont,
             const MatTimesVec& mattimesvec, const VecTimesVec& vectimesvec,
             const VecPlusVec& vecplusvec,
             const PolyPlusPolyPlusPoly& polypluspolypluspoly,
             const DecodeVec& decode_vec_t, const PolyFromMsg& poly_frommsg,
             const CompressVec& compress_vec, const CompressPoly& compress_poly)
      : nin(ninputs),
        generate_at(gen_mat_transposed),
        generate_r(gen_vec_r),
        generate_e1(gen_vec_e1),
        generate_e2(gen_poly),
        ntt_r(fwdntt_vec_r),
        intt_ar(invntt_vec_ar_tomont),
        intt_tr(invntt_poly_tr_tomont),
        a_times_r(mattimesvec),
        t_times_r(vectimesvec),
        ar_plus_e(vecplusvec),
        tr_plus_e2_plus_m(polypluspolypluspoly),
        decode_t(decode_vec_t),
        frommsg(poly_frommsg),
        compress_u(compress_vec),
        compress_v(compress_poly) {}
};

}  // namespace atpqc_cuda::kyber::primitive::cpapke_enc

#endif
