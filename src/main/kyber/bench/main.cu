//
// main.cu
// Measurement of primitive functions.
//
// Copyright (c) 2021 Tatsuki Ono
//
// This software is released under the MIT License.
// https://opensource.org/licenses/mit-license.php
//

#include <algorithm>
#include <array>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <vector>

#include "../../../lib/cuda_debug.hpp"
#include "../../../lib/cuda_resource.hpp"
#include "../../../lib/fips202_ws/host.cuh"
#include "../../../lib/kyber/arithmetic_mt/host.cuh"
#include "../../../lib/kyber/endecode_mt/host.cuh"
#include "../../../lib/kyber/genpoly_warp/host.cuh"
#include "../../../lib/kyber/ntt_ctgs_128t/host.cuh"
#include "../../../lib/kyber/ntt_ctgs_64t/host.cuh"
#include "../../../lib/kyber/params.cuh"
#include "../../../lib/kyber/primitive/ccakem_dec.cuh"
#include "../../../lib/kyber/primitive/ccakem_enc.cuh"
#include "../../../lib/kyber/primitive/ccakem_keypair.cuh"
#include "../../../lib/kyber/primitive/cpapke_dec.cuh"
#include "../../../lib/kyber/primitive/cpapke_enc.cuh"
#include "../../../lib/kyber/primitive/cpapke_keypair.cuh"
#include "../../../lib/kyber/symmetric_ws/host.cuh"
#include "../../../lib/kyber/variants.cuh"
#include "../../../lib/rng/std_random_device.hpp"
#include "../../../lib/verify_cmov_ws/host.cuh"

#ifndef KYBER_VARIANT
#define KYBER_VARIANT kyber512
#endif

namespace atpqc_cuda::kyber::bench {

using rng_type = rng::std_random_device;
using variant = variants::KYBER_VARIANT;
constexpr variant variant_v;
constexpr unsigned nloops = 4;

template <class Keypair, class Enc, class Dec>
std::tuple<double, double, double> measure(
    const Keypair& keypair, const Enc& enc, const Dec& dec, unsigned ninputs,
    const cuda_resource::device_pitched_memory<std::uint8_t>& pk_d,
    const cuda_resource::device_pitched_memory<std::uint8_t>& sk_d,
    const cuda_resource::device_pitched_memory<std::uint8_t>& ct_d,
    const cuda_resource::device_pitched_memory<std::uint8_t>& ss_d,
    const cuda_resource::pinned_memory<std::uint8_t>& pk_h,
    const cuda_resource::pinned_memory<std::uint8_t>& sk_h,
    const cuda_resource::pinned_memory<std::uint8_t>& ct_h,
    const cuda_resource::pinned_memory<std::uint8_t>& ss_h,
    const primitive::ccakem_keypair::mem_resource<variant>& keypair_mr,
    const primitive::ccakem_enc::mem_resource<variant>& enc_mr,
    const primitive::ccakem_dec::mem_resource<variant>& dec_mr) {
  cuda_resource::graph graph;

  cudaGraphNode_t begin_keypair_record;
  cuda_resource::event begin_keypair_event(cudaEventDefault);
  {
    CCC(cudaGraphAddEventRecordNode(&begin_keypair_record, graph, nullptr, 0,
                                    begin_keypair_event));
  }

  cudaGraphNode_t pk_available, sk_available;
  auto keypair_resource = keypair.join_graph(
      graph, pk_d.get_ptr(), pk_d.get_pitch(), begin_keypair_record,
      &pk_available, sk_d.get_ptr(), sk_d.get_pitch(), begin_keypair_record,
      &sk_available, keypair_mr);

  rng_type randombytes;
  randombytes(keypair_mr.pke_keypair_mr.rand_host.get_ptr(),
              params::symbytes * ninputs);
  randombytes(keypair_mr.rand_host.get_ptr(), params::symbytes * ninputs);

  cudaGraphNode_t cpypk_tohost_node;
  {
    cudaMemcpy3DParms memcpy_params;
    memcpy_params.srcPos = make_cudaPos(0, 0, 0);
    memcpy_params.srcPtr =
        make_cudaPitchedPtr(pk_d.get_ptr(), pk_d.get_pitch(),
                            params::publickeybytes<variant>, ninputs);
    memcpy_params.srcArray = nullptr;
    memcpy_params.dstPos = make_cudaPos(0, 0, 0);
    memcpy_params.dstPtr =
        make_cudaPitchedPtr(pk_h.get_ptr(), params::publickeybytes<variant>,
                            params::publickeybytes<variant>, ninputs);
    memcpy_params.dstArray = nullptr;
    memcpy_params.extent =
        make_cudaExtent(params::publickeybytes<variant>, ninputs, 1);
    memcpy_params.kind = cudaMemcpyDeviceToHost;
    std::array dep_array{pk_available};
    CCC(cudaGraphAddMemcpyNode(&cpypk_tohost_node, graph, dep_array.data(),
                               dep_array.size(), &memcpy_params));
  }

  cudaGraphNode_t cpysk_tohost_node;
  {
    cudaMemcpy3DParms memcpy_params;
    memcpy_params.srcPos = make_cudaPos(0, 0, 0);
    memcpy_params.srcPtr =
        make_cudaPitchedPtr(sk_d.get_ptr(), sk_d.get_pitch(),
                            params::secretkeybytes<variant>, ninputs);
    memcpy_params.srcArray = nullptr;
    memcpy_params.dstPos = make_cudaPos(0, 0, 0);
    memcpy_params.dstPtr =
        make_cudaPitchedPtr(sk_h.get_ptr(), params::secretkeybytes<variant>,
                            params::secretkeybytes<variant>, ninputs);
    memcpy_params.dstArray = nullptr;
    memcpy_params.extent =
        make_cudaExtent(params::secretkeybytes<variant>, ninputs, 1);
    memcpy_params.kind = cudaMemcpyDeviceToHost;
    std::array dep_array{sk_available};
    CCC(cudaGraphAddMemcpyNode(&cpysk_tohost_node, graph, dep_array.data(),
                               dep_array.size(), &memcpy_params));
  }

  cudaGraphNode_t keypair_enc_record;
  cuda_resource::event keypair_enc_event(cudaEventDefault);
  {
    std::array dep_array{cpypk_tohost_node, cpysk_tohost_node};
    CCC(cudaGraphAddEventRecordNode(&keypair_enc_record, graph,
                                    dep_array.data(), dep_array.size(),
                                    keypair_enc_event));
  }

  cudaGraphNode_t cpypk_todev_node;
  {
    cudaMemcpy3DParms memcpy_params;
    memcpy_params.srcPos = make_cudaPos(0, 0, 0);
    memcpy_params.srcPtr =
        make_cudaPitchedPtr(pk_h.get_ptr(), params::publickeybytes<variant>,
                            params::publickeybytes<variant>, ninputs);
    memcpy_params.srcArray = nullptr;
    memcpy_params.dstPos = make_cudaPos(0, 0, 0);
    memcpy_params.dstPtr =
        make_cudaPitchedPtr(pk_d.get_ptr(), pk_d.get_pitch(),
                            params::publickeybytes<variant>, ninputs);
    memcpy_params.dstArray = nullptr;
    memcpy_params.extent =
        make_cudaExtent(params::publickeybytes<variant>, ninputs, 1);
    memcpy_params.kind = cudaMemcpyHostToDevice;
    std::array dep_array{keypair_enc_record};
    CCC(cudaGraphAddMemcpyNode(&cpypk_todev_node, graph, dep_array.data(),
                               dep_array.size(), &memcpy_params));
  }

  cudaGraphNode_t ct_available, ssb_available, pk_used;
  auto enc_resource = enc.join_graph(
      graph, ct_d.get_ptr(), ct_d.get_pitch(), keypair_enc_record,
      &ct_available, ss_d.get_ptr(), ss_d.get_pitch(), keypair_enc_record,
      &ssb_available, pk_d.get_ptr(), pk_d.get_pitch(), cpypk_todev_node,
      &pk_used, enc_mr);

  randombytes(enc_mr.rand_host.get_ptr(), params::symbytes * ninputs);

  cudaGraphNode_t cpyct_tohost_node;
  {
    cudaMemcpy3DParms memcpy_params;
    memcpy_params.srcPos = make_cudaPos(0, 0, 0);
    memcpy_params.srcPtr =
        make_cudaPitchedPtr(ct_d.get_ptr(), ct_d.get_pitch(),
                            params::ciphertextbytes<variant>, ninputs);
    memcpy_params.srcArray = nullptr;
    memcpy_params.dstPos = make_cudaPos(0, 0, 0);
    memcpy_params.dstPtr =
        make_cudaPitchedPtr(ct_h.get_ptr(), params::ciphertextbytes<variant>,
                            params::ciphertextbytes<variant>, ninputs);
    memcpy_params.dstArray = nullptr;
    memcpy_params.extent =
        make_cudaExtent(params::ciphertextbytes<variant>, ninputs, 1);
    memcpy_params.kind = cudaMemcpyDeviceToHost;
    std::array dep_array{ct_available};
    CCC(cudaGraphAddMemcpyNode(&cpyct_tohost_node, graph, dep_array.data(),
                               dep_array.size(), &memcpy_params));
  }

  cudaGraphNode_t cpyssb_tohost_node;
  {
    cudaMemcpy3DParms memcpy_params;
    memcpy_params.srcPos = make_cudaPos(0, 0, 0);
    memcpy_params.srcPtr = make_cudaPitchedPtr(ss_d.get_ptr(), ss_d.get_pitch(),
                                               params::ssbytes, ninputs);
    memcpy_params.srcArray = nullptr;
    memcpy_params.dstPos = make_cudaPos(0, 0, 0);
    memcpy_params.dstPtr = make_cudaPitchedPtr(ss_h.get_ptr(), params::ssbytes,
                                               params::ssbytes, ninputs);
    memcpy_params.dstArray = nullptr;
    memcpy_params.extent = make_cudaExtent(params::ssbytes, ninputs, 1);
    memcpy_params.kind = cudaMemcpyDeviceToHost;
    std::array dep_array{ssb_available};
    CCC(cudaGraphAddMemcpyNode(&cpyssb_tohost_node, graph, dep_array.data(),
                               dep_array.size(), &memcpy_params));
  }

  cudaGraphNode_t enc_dec_record;
  cuda_resource::event enc_dec_event(cudaEventDefault);
  {
    std::array dep_array{cpyct_tohost_node, cpyssb_tohost_node};
    CCC(cudaGraphAddEventRecordNode(&enc_dec_record, graph, dep_array.data(),
                                    dep_array.size(), enc_dec_event));
  }

  cudaGraphNode_t cpysk_todev_node;
  {
    cudaMemcpy3DParms memcpy_params;
    memcpy_params.srcPos = make_cudaPos(0, 0, 0);
    memcpy_params.srcPtr =
        make_cudaPitchedPtr(sk_h.get_ptr(), params::secretkeybytes<variant>,
                            params::secretkeybytes<variant>, ninputs);
    memcpy_params.srcArray = nullptr;
    memcpy_params.dstPos = make_cudaPos(0, 0, 0);
    memcpy_params.dstPtr =
        make_cudaPitchedPtr(sk_d.get_ptr(), sk_d.get_pitch(),
                            params::secretkeybytes<variant>, ninputs);
    memcpy_params.dstArray = nullptr;
    memcpy_params.extent =
        make_cudaExtent(params::secretkeybytes<variant>, ninputs, 1);
    memcpy_params.kind = cudaMemcpyHostToDevice;
    std::array dep_array{enc_dec_record};
    CCC(cudaGraphAddMemcpyNode(&cpysk_todev_node, graph, dep_array.data(),
                               dep_array.size(), &memcpy_params));
  }

  cudaGraphNode_t cpyct_todev_node;
  {
    cudaMemcpy3DParms memcpy_params;
    memcpy_params.srcPos = make_cudaPos(0, 0, 0);
    memcpy_params.srcPtr =
        make_cudaPitchedPtr(ct_h.get_ptr(), params::ciphertextbytes<variant>,
                            params::ciphertextbytes<variant>, ninputs);
    memcpy_params.srcArray = nullptr;
    memcpy_params.dstPos = make_cudaPos(0, 0, 0);
    memcpy_params.dstPtr =
        make_cudaPitchedPtr(ct_d.get_ptr(), ct_d.get_pitch(),
                            params::ciphertextbytes<variant>, ninputs);
    memcpy_params.dstArray = nullptr;
    memcpy_params.extent =
        make_cudaExtent(params::ciphertextbytes<variant>, ninputs, 1);
    memcpy_params.kind = cudaMemcpyHostToDevice;
    std::array dep_array{enc_dec_record};
    CCC(cudaGraphAddMemcpyNode(&cpyct_todev_node, graph, dep_array.data(),
                               dep_array.size(), &memcpy_params));
  }

  cudaGraphNode_t ssa_available, sk_used, ct_used;
  auto dec_resource = dec.join_graph(
      graph, ss_d.get_ptr(), ss_d.get_pitch(), enc_dec_record, &ssa_available,
      ct_d.get_ptr(), ct_d.get_pitch(), cpyct_todev_node, &ct_used,
      sk_d.get_ptr(), sk_d.get_pitch(), cpysk_todev_node, &sk_used, dec_mr);

  cudaGraphNode_t cpyssa_tohost_node;
  {
    cudaMemcpy3DParms memcpy_params;
    memcpy_params.srcPos = make_cudaPos(0, 0, 0);
    memcpy_params.srcPtr = make_cudaPitchedPtr(ss_d.get_ptr(), ss_d.get_pitch(),
                                               params::ssbytes, ninputs);
    memcpy_params.srcArray = nullptr;
    memcpy_params.dstPos = make_cudaPos(0, 0, 0);
    memcpy_params.dstPtr = make_cudaPitchedPtr(ss_h.get_ptr(), params::ssbytes,
                                               params::ssbytes, ninputs);
    memcpy_params.dstArray = nullptr;
    memcpy_params.extent = make_cudaExtent(params::ssbytes, ninputs, 1);
    memcpy_params.kind = cudaMemcpyDeviceToHost;
    std::array dep_array{ssa_available};
    CCC(cudaGraphAddMemcpyNode(&cpyssa_tohost_node, graph, dep_array.data(),
                               dep_array.size(), &memcpy_params));
  }

  cudaGraphNode_t dec_end_record;
  cuda_resource::event dec_end_event(cudaEventDefault);
  {
    std::array dep_array{cpyssa_tohost_node};
    CCC(cudaGraphAddEventRecordNode(&dec_end_record, graph, dep_array.data(),
                                    dep_array.size(), dec_end_event));
  }

  cuda_resource::graph_exec exec(graph);

  cuda_resource::stream stream(cudaStreamNonBlocking);

  std::vector<float> raw_keypair_time(nloops);
  std::vector<float> raw_enc_time(nloops);
  std::vector<float> raw_dec_time(nloops);

  CCC(cudaGraphLaunch(exec, stream));
  CCC(cudaStreamSynchronize(stream));
  for (unsigned i = 0; i < nloops; ++i) {
    CCC(cudaGraphLaunch(exec, stream));
    CCC(cudaStreamSynchronize(stream));

    CCC(cudaEventElapsedTime(&raw_keypair_time.at(i), begin_keypair_event,
                             keypair_enc_event));
    CCC(cudaEventElapsedTime(&raw_enc_time.at(i), keypair_enc_event,
                             enc_dec_event));
    CCC(cudaEventElapsedTime(&raw_dec_time.at(i), enc_dec_event,
                             dec_end_event));
  }

  auto accfn = [](double a, double b) noexcept -> double { return a + b; };
  double keypair_latency = std::accumulate(raw_keypair_time.begin(),
                                           raw_keypair_time.end(), 0.0, accfn) /
                           nloops * 1e-3;
  double enc_latency =
      std::accumulate(raw_enc_time.begin(), raw_enc_time.end(), 0.0, accfn) /
      nloops * 1e-3;
  double dec_latency =
      std::accumulate(raw_dec_time.begin(), raw_dec_time.end(), 0.0, accfn) /
      nloops * 1e-3;

  return {keypair_latency, enc_latency, dec_latency};
}

void bench() {
  unsigned ninputs;
  unsigned genmat_nwarps;
  unsigned genvec_nwarps;
  unsigned genpoly_nwarps;
  unsigned fips202_nwarps;

  std::cin >> ninputs >> genmat_nwarps >> genvec_nwarps >> genpoly_nwarps >>
      fips202_nwarps;

  primitive::ccakem_keypair::mem_resource<variant> keypair_mr(ninputs);
  primitive::ccakem_enc::mem_resource<variant> enc_mr(ninputs);
  primitive::ccakem_dec::mem_resource<variant> dec_mr(ninputs);

  cuda_resource::device_pitched_memory<std::uint8_t> pk_d(
      params::publickeybytes<variant>, ninputs);
  cuda_resource::device_pitched_memory<std::uint8_t> sk_d(
      params::secretkeybytes<variant>, ninputs);
  cuda_resource::device_pitched_memory<std::uint8_t> ct_d(
      params::ciphertextbytes<variant>, ninputs);
  cuda_resource::device_pitched_memory<std::uint8_t> ss_d(params::ssbytes,
                                                          ninputs);

  cuda_resource::pinned_memory<std::uint8_t> pk_h(
      params::publickeybytes<variant> * ninputs);
  cuda_resource::pinned_memory<std::uint8_t> sk_h(
      params::secretkeybytes<variant> * ninputs);
  cuda_resource::pinned_memory<std::uint8_t> ct_h(
      params::ciphertextbytes<variant> * ninputs);
  cuda_resource::pinned_memory<std::uint8_t> ss_h(params::ssbytes * ninputs);

  rng_type randombytes;
  symmetric_ws::host::hash_g hash_seed(ninputs, fips202_nwarps);
  genpoly_warp::host::gena<params::k<variant>> generate_a(ninputs,
                                                          genmat_nwarps);
  genpoly_warp::host::genat<params::k<variant>> generate_at(ninputs,
                                                            genmat_nwarps);
  genpoly_warp::host::gennoise<params::k<variant>, params::eta1<variant>>
      generate_s(ninputs, genvec_nwarps);
  genpoly_warp::host::gennoise<params::k<variant>, params::eta1<variant>>
      generate_e(ninputs, genvec_nwarps);
  genpoly_warp::host::gennoise<params::k<variant>, params::eta1<variant>>
      generate_r(ninputs, genvec_nwarps);
  genpoly_warp::host::gennoise<params::k<variant>, params::eta2> generate_e1(
      ninputs, genvec_nwarps);
  genpoly_warp::host::gennoise<1, params::eta2> generate_e2(ninputs,
                                                            genpoly_nwarps);
  ntt_ctgs_64t::host::fwdntt<params::k<variant>> fwdnttvec_s(ninputs);
  ntt_ctgs_64t::host::fwdntt<params::k<variant>> fwdnttvec_e(ninputs);
  ntt_ctgs_64t::host::fwdntt<params::k<variant>> fwdnttvec_r(ninputs);
  ntt_ctgs_64t::host::fwdntt<params::k<variant>> fwdnttvec_u(ninputs);
  ntt_ctgs_64t::host::invntt_tomont<params::k<variant>> intt_ar(ninputs);
  ntt_ctgs_64t::host::invntt_tomont<1> intt_tr(ninputs);
  ntt_ctgs_64t::host::invntt_tomont<1> intt_su(ninputs);
  arithmetic_mt::host::mattimesvec_tomont_plusvec<params::k<variant>> mtvpv(
      ninputs);
  arithmetic_mt::host::mattimesvec<params::k<variant>> mtv(ninputs);
  arithmetic_mt::host::vectimesvec<params::k<variant>> ttimesr(ninputs);
  arithmetic_mt::host::vectimesvec<params::k<variant>> stimesu(ninputs);
  arithmetic_mt::host::vecadd2<params::k<variant>> vpv(ninputs);
  arithmetic_mt::host::polyadd3 padd3(ninputs);
  arithmetic_mt::host::polysub psub(ninputs);
  endecode_mt::host::polyvec_tobytes<params::k<variant>> encodet(ninputs);
  endecode_mt::host::polyvec_tobytes<params::k<variant>> encodes(ninputs);
  endecode_mt::host::polyvec_frombytes<params::k<variant>> decodet(ninputs);
  endecode_mt::host::polyvec_frombytes<params::k<variant>> decodes(ninputs);
  endecode_mt::host::poly_frommsg frommsg(ninputs);
  endecode_mt::host::poly_tomsg tomsg(ninputs);
  endecode_mt::host::polyvec_compress<params::k<variant>, params::du<variant>>
      compressu(ninputs);
  endecode_mt::host::poly_compress<params::dv<variant>> compressv(ninputs);
  endecode_mt::host::polyvec_decompress<params::k<variant>, params::du<variant>>
      decompressu(ninputs);
  endecode_mt::host::poly_decompress<params::dv<variant>> decompressv(ninputs);

  symmetric_ws::host::hash_h keypair_hash_pk(ninputs, fips202_nwarps);
  symmetric_ws::host::hash_h enc_hash_rand(ninputs, fips202_nwarps);
  symmetric_ws::host::hash_h enc_hash_pk(ninputs, fips202_nwarps);
  symmetric_ws::host::hash_h enc_hash_ct(ninputs, fips202_nwarps);
  symmetric_ws::host::hash_g enc_hash_coin(ninputs, fips202_nwarps);
  symmetric_ws::host::kdf enc_kdf(ninputs, fips202_nwarps);
  symmetric_ws::host::hash_h dec_hash_ct(ninputs, fips202_nwarps);
  symmetric_ws::host::hash_g dec_hash_coin(ninputs, fips202_nwarps);
  symmetric_ws::host::kdf dec_kdf(ninputs, fips202_nwarps);
  verify_cmov_ws::host::verify_cmov dec_verify_cmov(ninputs);

  primitive::ccakem_keypair::keypair keypair(
      ninputs, variant_v,
      primitive::cpapke_keypair::cpapke_keypair(
          ninputs, variant_v, randombytes, hash_seed, generate_a, generate_s,
          generate_e, fwdnttvec_s, fwdnttvec_e, mtvpv, encodet, encodes),
      randombytes, keypair_hash_pk);
  primitive::ccakem_enc::enc enc(
      ninputs, variant_v,
      primitive::cpapke_enc::cpapke_enc(
          ninputs, variant_v, generate_at, generate_r, generate_e1, generate_e2,
          fwdnttvec_r, intt_ar, intt_tr, mtv, ttimesr, vpv, padd3, decodet,
          frommsg, compressu, compressv),
      randombytes, enc_hash_rand, enc_hash_pk, enc_hash_ct, enc_hash_coin,
      enc_kdf);
  primitive::ccakem_dec::dec dec(
      ninputs, variant_v,
      primitive::cpapke_enc::cpapke_enc(
          ninputs, variant_v, generate_at, generate_r, generate_e1, generate_e2,
          fwdnttvec_r, intt_ar, intt_tr, mtv, ttimesr, vpv, padd3, decodet,
          frommsg, compressu, compressv),
      primitive::cpapke_dec::cpapke_dec(ninputs, variant_v, fwdnttvec_u,
                                        intt_su, stimesu, psub, decodes,
                                        decompressu, decompressv, tomsg),
      dec_hash_ct, dec_hash_coin, dec_kdf, dec_verify_cmov);

  auto [keypair_latency, enc_latency, dec_latency] =
      measure(keypair, enc, dec, ninputs, pk_d, sk_d, ct_d, ss_d, pk_h, sk_h,
              ct_h, ss_h, keypair_mr, enc_mr, dec_mr);
  double keypair_throughput = ninputs / keypair_latency;
  double enc_throughput = ninputs / enc_latency;
  double dec_throughput = ninputs / dec_latency;

  std::printf("%le,%le,%le,%le,%le,%le\n", keypair_latency, keypair_throughput,
              enc_latency, enc_throughput, dec_latency, dec_throughput);
}

}  // namespace atpqc_cuda::kyber::bench

int main() {
  CUDA_DEBUG_RESET();

  CCC(cuInit(0));
  CUdevice dev;
  CCC(cuDeviceGet(&dev, 0));

  {
    atpqc_cuda::cuda_resource::context ctx(dev);

    { atpqc_cuda::kyber::bench::bench(); }

    CCC(cuCtxSynchronize());
  }

  return 0;
}
