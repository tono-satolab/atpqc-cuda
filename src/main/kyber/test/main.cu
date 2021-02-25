//
// main.cu
// Testing primitive functions.
//
// Copyright (c) 2021 Tatsuki Ono
//
// This software is released under the MIT License.
// https://opensource.org/licenses/mit-license.php
//

#include <cstddef>
#include <cstdio>
#include <cstring>

#include "../../../lib/fips202_ws/host.cuh"
#include "../../../lib/kyber/arithmetic_mt/host.cuh"
#include "../../../lib/kyber/endecode_mt/host.cuh"
#include "../../../lib/kyber/genpoly_warp/host.cuh"
#include "../../../lib/kyber/ntt_ctgs_64t/host.cuh"
#include "../../../lib/kyber/primitive/ccakem_dec.cuh"
#include "../../../lib/kyber/primitive/ccakem_enc.cuh"
#include "../../../lib/kyber/primitive/ccakem_keypair.cuh"
#include "../../../lib/kyber/primitive/cpapke_dec.cuh"
#include "../../../lib/kyber/primitive/cpapke_enc.cuh"
#include "../../../lib/kyber/primitive/cpapke_keypair.cuh"
#include "../../../lib/kyber/symmetric_ws/host.cuh"
#include "../../../lib/kyber/variants.cuh"
#include "../../../lib/rng/std_mt19937_64.hpp"
#include "../../../lib/rng/std_random_device.hpp"
#include "../../../lib/rng/zero.hpp"
#include "../../../lib/verify_cmov_ws/host.cuh"

#ifndef KYBER_VARIANT
#define KYBER_VARIANT kyber512
#endif

#ifndef RNG_KIND
#define RNG_KIND std_random_device
#endif

namespace atpqc_cuda::kyber::test {

constexpr unsigned ninputs = 128;
constexpr unsigned ntests = 16;
using variant = kyber::variants::KYBER_VARIANT;
using rng_type = rng::RNG_KIND;
constexpr variant variant_v;
constexpr unsigned genmatrix_nwarps = params::k<variant> * params::k<variant>;
constexpr unsigned genvec_nwarps = params::k<variant>;
constexpr unsigned genpoly_nwarps = 1;
constexpr unsigned fips_nwarps = 1;

int test_kyber() {
  rng_type randombytes;
  symmetric_ws::host::hash_g hash_seed(ninputs, fips_nwarps);
  genpoly_warp::host::gena<params::k<variant>> generate_a(ninputs,
                                                          genmatrix_nwarps);
  genpoly_warp::host::genat<params::k<variant>> generate_at(ninputs,
                                                            genmatrix_nwarps);
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

  primitive::cpapke_keypair::cpapke_keypair pke_keypair(
      ninputs, variant_v, randombytes, hash_seed, generate_a, generate_s,
      generate_e, fwdnttvec_s, fwdnttvec_e, mtvpv, encodet, encodes);
  primitive::cpapke_enc::cpapke_enc pke_enc(
      ninputs, variant_v, generate_at, generate_r, generate_e1, generate_e2,
      fwdnttvec_r, intt_ar, intt_tr, mtv, ttimesr, vpv, padd3, decodet, frommsg,
      compressu, compressv);
  primitive::cpapke_dec::cpapke_dec pke_dec(ninputs, variant_v, fwdnttvec_u,
                                            intt_su, stimesu, psub, decodes,
                                            decompressu, decompressv, tomsg);

  rng_type keypair_randombytes;
  symmetric_ws::host::hash_h keypair_hash_pk(ninputs, fips_nwarps);
  rng_type enc_randombytes;
  symmetric_ws::host::hash_h enc_hash_rand(ninputs, fips_nwarps);
  symmetric_ws::host::hash_h enc_hash_pk(ninputs, fips_nwarps);
  symmetric_ws::host::hash_h enc_hash_ct(ninputs, fips_nwarps);
  symmetric_ws::host::hash_g enc_hash_coin(ninputs, fips_nwarps);
  symmetric_ws::host::kdf enc_kdf(ninputs, fips_nwarps);
  symmetric_ws::host::hash_h dec_hash_ct(ninputs, fips_nwarps);
  symmetric_ws::host::hash_g dec_hash_coin(ninputs, fips_nwarps);
  symmetric_ws::host::kdf dec_kdf(ninputs, fips_nwarps);
  verify_cmov_ws::host::verify_cmov dec_verify_cmov(ninputs);

  primitive::ccakem_keypair::keypair keypair(
      ninputs, variant_v,
      primitive::cpapke_keypair::cpapke_keypair(
          ninputs, variant_v, randombytes, hash_seed, generate_a, generate_s,
          generate_e, fwdnttvec_s, fwdnttvec_e, mtvpv, encodet, encodes),
      keypair_randombytes, keypair_hash_pk);
  primitive::ccakem_enc::enc enc(
      ninputs, variant_v,
      primitive::cpapke_enc::cpapke_enc(
          ninputs, variant_v, generate_at, generate_r, generate_e1, generate_e2,
          fwdnttvec_r, intt_ar, intt_tr, mtv, ttimesr, vpv, padd3, decodet,
          frommsg, compressu, compressv),
      enc_randombytes, enc_hash_rand, enc_hash_pk, enc_hash_ct, enc_hash_coin,
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

  {
    cuda_resource::device_pitched_memory<std::uint8_t> pk_d(
        params::publickeybytes<variant>, ninputs);
    cuda_resource::device_pitched_memory<std::uint8_t> sk_d(
        params::secretkeybytes<variant>, ninputs);
    cuda_resource::device_pitched_memory<std::uint8_t> ct_d(
        params::ciphertextbytes<variant>, ninputs);
    cuda_resource::device_pitched_memory<std::uint8_t> key_a_d(params::ssbytes,
                                                               ninputs);
    cuda_resource::device_pitched_memory<std::uint8_t> key_b_d(params::ssbytes,
                                                               ninputs);

    cuda_resource::pinned_memory<std::uint8_t> pinned_key_a(
        params::ssbytes * ninputs, cudaHostAllocDefault);
    cuda_resource::pinned_memory<std::uint8_t> pinned_key_b(
        params::ssbytes * ninputs, cudaHostAllocDefault);

    cuda_resource::graph graph;
    cudaGraphNode_t empty_root, dummy_node, keypair_pk, keypair_sk;
    cudaGraphNode_t enc_ct, enc_key;
    cudaGraphNode_t dec_key;
    std::array<std::uint8_t*, 2> key_ptrs{key_a_d.get_ptr(), key_b_d.get_ptr()};

    cuda_resource::stream stream0(cudaStreamNonBlocking);

    class key_cmp {
     public:
      using data_type = std::uint8_t**;
      static void cmp(void* ptr) {
        data_type dptr = static_cast<data_type>(ptr);
        std::uint8_t* key_a = dptr[0];
        std::uint8_t* key_b = dptr[1];

        for (unsigned j = 0; j < ninputs; ++j) {
          if (memcmp(key_a + params::ssbytes * j, key_b + params::ssbytes * j,
                     params::ssbytes)) {
            std::printf("parallel: %u, ERROR keys\n", j);

          } else {
            // std::printf("parallel: %u, SUCCESS keys\n", j);
          }
          // {
          //   std::printf("key_a: ");
          //   for (unsigned i0 = 0; i0 < params::ssbytes; ++i0) {
          //     std::printf("%02x ", key_a[j * params::ssbytes + i0]);
          //   }
          //   std::printf("\n");
          //   std::printf("key_b: ");
          //   for (unsigned i0 = 0; i0 < params::ssbytes; ++i0) {
          //     std::printf("%02x ", key_b[j * params::ssbytes + i0]);
          //   }
          //   std::printf("\n");
          // }  // debug
        }
      }
    };

    primitive::ccakem_keypair::mem_resource<variant> keypair_mem(ninputs);
    primitive::ccakem_enc::mem_resource<variant> enc_mem(ninputs);
    primitive::ccakem_dec::mem_resource<variant> dec_mem(ninputs);

    CCC(cudaGraphAddEmptyNode(&empty_root, graph, nullptr, 0));
    auto keypair_args = keypair.join_graph(
        graph, pk_d.get_ptr(), pk_d.get_pitch(), empty_root, &keypair_pk,
        sk_d.get_ptr(), sk_d.get_pitch(), empty_root, &keypair_sk, keypair_mem);
    auto enc_args = enc.join_graph(
        graph, ct_d.get_ptr(), ct_d.get_pitch(), empty_root, &enc_ct,
        key_b_d.get_ptr(), key_b_d.get_pitch(), empty_root, &enc_key,
        pk_d.get_ptr(), pk_d.get_pitch(), keypair_pk, &dummy_node, enc_mem);
    auto dec_args = dec.join_graph(
        graph, key_a_d.get_ptr(), key_a_d.get_pitch(), empty_root, &dec_key,
        ct_d.get_ptr(), ct_d.get_pitch(), enc_ct, &dummy_node, sk_d.get_ptr(),
        sk_d.get_pitch(), keypair_sk, &dummy_node, dec_mem);

    randombytes(keypair_mem.pke_keypair_mr.rand_host.get_ptr(),
                params::symbytes * ninputs);
    randombytes(keypair_mem.rand_host.get_ptr(), params::symbytes * ninputs);
    randombytes(enc_mem.rand_host.get_ptr(), params::symbytes * ninputs);

    cudaGraphNode_t cpykeya_node;
    {
      cudaMemcpy3DParms memcpy_params;
      memcpy_params.srcPos = make_cudaPos(0, 0, 0);
      memcpy_params.srcPtr = make_cudaPitchedPtr(
          key_a_d.get_ptr(), key_a_d.get_pitch(), params::ssbytes, ninputs);
      memcpy_params.srcArray = nullptr;
      memcpy_params.dstPos = make_cudaPos(0, 0, 0);
      memcpy_params.dstPtr = make_cudaPitchedPtr(
          pinned_key_a.get_ptr(), params::ssbytes, params::ssbytes, ninputs);
      memcpy_params.dstArray = nullptr;
      memcpy_params.extent = make_cudaExtent(params::ssbytes, ninputs, 1);
      memcpy_params.kind = cudaMemcpyDeviceToHost;
      std::array dep_array{dec_key};
      CCC(cudaGraphAddMemcpyNode(&cpykeya_node, graph, dep_array.data(),
                                 dep_array.size(), &memcpy_params));
    }

    cudaGraphNode_t cpykeyb_node;
    {
      cudaMemcpy3DParms memcpy_params;
      memcpy_params.srcPos = make_cudaPos(0, 0, 0);
      memcpy_params.srcPtr = make_cudaPitchedPtr(
          key_b_d.get_ptr(), key_b_d.get_pitch(), params::ssbytes, ninputs);
      memcpy_params.srcArray = nullptr;
      memcpy_params.dstPos = make_cudaPos(0, 0, 0);
      memcpy_params.dstPtr = make_cudaPitchedPtr(
          pinned_key_b.get_ptr(), params::ssbytes, params::ssbytes, ninputs);
      memcpy_params.dstArray = nullptr;
      memcpy_params.extent = make_cudaExtent(params::ssbytes, ninputs, 1);
      memcpy_params.kind = cudaMemcpyDeviceToHost;
      std::array dep_array{enc_key};
      CCC(cudaGraphAddMemcpyNode(&cpykeyb_node, graph, dep_array.data(),
                                 dep_array.size(), &memcpy_params));
    }

    std::array cmp_ptr_array{pinned_key_a.get_ptr(), pinned_key_b.get_ptr()};
    void* vptr = cmp_ptr_array.data();
    std::array cmp_dep_array{cpykeya_node, cpykeyb_node};
    cudaGraphNode_t randombytes_node;
    cudaHostNodeParams host_params{key_cmp::cmp, vptr};
    CCC(cudaGraphAddHostNode(&randombytes_node, graph, cmp_dep_array.data(),
                             cmp_dep_array.size(), &host_params));

    cuda_resource::graph_exec exec(graph, &dummy_node, nullptr, 0);

    for (unsigned i = 0; i < ntests; ++i) {
      std::printf("loop: %05u\n", i);

      CCC(cudaGraphLaunch(exec, stream0));
      CCC(cudaStreamSynchronize(stream0));
    }

    // {
    //   std::uint8_t pk_h[ninputs][params::publickeybytes<variant>];
    //   std::uint8_t sk_h[ninputs][params::secretkeybytes<variant>];

    //   CCC(cudaMemcpy2D(pk_h, params::publickeybytes<variant>,
    //   pk_d.get_ptr(),
    //                    pk_d.get_pitch(), params::publickeybytes<variant>,
    //                    ninputs, cudaMemcpyDeviceToHost));
    //   CCC(cudaMemcpy2D(sk_h, params::secretkeybytes<variant>,
    //   sk_d.get_ptr(),
    //                    sk_d.get_pitch(), params::secretkeybytes<variant>,
    //                    ninputs, cudaMemcpyDeviceToHost));

    //   std::printf("pk:\n");
    //   for (unsigned j = 0; j < ninputs; ++j) {
    //     std::printf("ID: %03u\n", j);
    //     for (unsigned k = 0; k < params::publickeybytes<variant>; ++k) {
    //       std::printf("%02x ", pk_h[j][k]);
    //       if (k % 32 == 31) std::printf("\n");
    //     }
    //     std::printf("\n");
    //   }
    //   std::printf("\n");

    //   std::printf("sk:\n");
    //   for (unsigned j = 0; j < ninputs; ++j) {
    //     std::printf("ID: %03u\n", j);
    //     for (unsigned k = 0; k < params::secretkeybytes<variant>; ++k) {
    //       std::printf("%02x ", sk_h[j][k]);
    //       if (k % 32 == 31) std::printf("\n");
    //     }
    //     std::printf("\n");
    //   }
    //   std::printf("\n");
    // }  // debug

    // {
    //   std::uint8_t ct_h[ninputs][params::ciphertextbytes<variant>];

    //   CCC(cudaMemcpy2D(ct_h, params::ciphertextbytes<variant>,
    //   ct_d.get_ptr(),
    //                    ct_d.get_pitch(), params::ciphertextbytes<variant>,
    //                    ninputs, cudaMemcpyDeviceToHost));

    //   std::printf("c:\n");
    //   for (unsigned j = 0; j < ninputs; ++j) {
    //     std::printf("ID: %03u\n", j);
    //     for (unsigned k = 0; k < params::ciphertextbytes<variant>; ++k) {
    //       std::printf("%02x ", ct_h[j][k]);
    //       if (k % 32 == 31) std::printf("\n");
    //     }
    //     std::printf("\n");
    //   }
    //   std::printf("\n");
    // }  // debug

  }  // ccakem

  return 0;
}

}  // namespace atpqc_cuda::kyber::test

int main(void) {
  CUDA_DEBUG_RESET();

  CCC(cuInit(0));
  CUdevice dev;
  CCC(cuDeviceGet(&dev, 0));

  {
    atpqc_cuda::cuda_resource::context ctx(dev);

    {
      using namespace atpqc_cuda::kyber;

      test::test_kyber();

      std::printf("CRYPTO_SECRETKEYBYTES:  %d\n",
                  params::secretkeybytes<test::variant>);
      std::printf("CRYPTO_PUBLICKEYBYTES:  %d\n",
                  params::publickeybytes<test::variant>);
      std::printf("CRYPTO_CIPHERTEXTBYTES: %d\n",
                  params::ciphertextbytes<test::variant>);
    }

    CCC(cuCtxSynchronize());
  }

  return 0;
}
