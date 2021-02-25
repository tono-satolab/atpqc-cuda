//
// params.cuh
// Parameters of Kyber.
//
// Copyright (c) 2021 Tatsuki Ono
//
// This software is released under the MIT License.
// https://opensource.org/licenses/mit-license.php
//

#ifndef ATPQC_CUDA_LIB_KYBER_PARAMS_CUH_
#define ATPQC_CUDA_LIB_KYBER_PARAMS_CUH_

#include <type_traits>

#include "variants.cuh"

namespace atpqc_cuda::kyber {

namespace params_type {

/* Degree of polynomial */
struct n_type : std::integral_constant<unsigned, 256> {};

/* Dimension of module */
template <class Variant>
struct k_type {};
template <>
struct k_type<variants::kyber512> : std::integral_constant<unsigned, 2> {};
template <>
struct k_type<variants::kyber768> : std::integral_constant<unsigned, 3> {};
template <>
struct k_type<variants::kyber1024> : std::integral_constant<unsigned, 4> {};

/* Modulus */
struct q_type : std::integral_constant<unsigned, 3329> {};

/* Used in noise sampling */
template <class Variant>
struct eta1_type {};
template <>
struct eta1_type<variants::kyber512> : std::integral_constant<unsigned, 3> {};
template <>
struct eta1_type<variants::kyber768> : std::integral_constant<unsigned, 2> {};
template <>
struct eta1_type<variants::kyber1024> : std::integral_constant<unsigned, 2> {};

struct eta2_type : std::integral_constant<unsigned, 2> {};

template <class Variant>
struct du_type {};
template <>
struct du_type<variants::kyber512> : std::integral_constant<unsigned, 10> {};
template <>
struct du_type<variants::kyber768> : std::integral_constant<unsigned, 10> {};
template <>
struct du_type<variants::kyber1024> : std::integral_constant<unsigned, 11> {};

template <class Variant>
struct dv_type {};
template <>
struct dv_type<variants::kyber512> : std::integral_constant<unsigned, 4> {};
template <>
struct dv_type<variants::kyber768> : std::integral_constant<unsigned, 4> {};
template <>
struct dv_type<variants::kyber1024> : std::integral_constant<unsigned, 5> {};

/* Size in bytes of hashes, and seeds */
struct symbytes_type : std::integral_constant<unsigned, 32> {};
/* Size in bytes of shared key */
struct ssbytes_type : std::integral_constant<unsigned, 32> {};

struct polybytes_type : std::integral_constant<unsigned, 384> {};
template <class Variant>
struct polyvecbytes_type
    : std::integral_constant<unsigned,
                             k_type<Variant>::value * polybytes_type::value> {};

template <class Variant>
struct polycompressedbytes_type
    : std::integral_constant<unsigned, 32 * dv_type<Variant>::value> {};
template <class Variant>
struct polyveccompressedbytes_type
    : std::integral_constant<unsigned, k_type<Variant>::value * 32 *
                                           du_type<Variant>::value> {};

/* INDCPA-PKE params */
struct indcpa_msgbytes_type
    : std::integral_constant<unsigned, symbytes_type::value> {};
template <class Variant>
struct indcpa_publickeybytes_type
    : std::integral_constant<unsigned, polyvecbytes_type<Variant>::value +
                                           symbytes_type::value> {};
template <class Variant>
struct indcpa_secretkeybytes_type
    : std::integral_constant<unsigned, polyvecbytes_type<Variant>::value> {};
template <class Variant>
struct indcpa_bytes_type
    : std::integral_constant<unsigned,
                             polyveccompressedbytes_type<Variant>::value +
                                 polycompressedbytes_type<Variant>::value> {};

/* INDCCA-KEM params */
template <class Variant>
struct publickeybytes_type
    : std::integral_constant<unsigned,
                             indcpa_publickeybytes_type<Variant>::value> {};
/* 32 bytes of additional space to save H(pk) */
template <class Variant>
struct secretkeybytes_type
    : std::integral_constant<unsigned,
                             indcpa_secretkeybytes_type<Variant>::value +
                                 indcpa_publickeybytes_type<Variant>::value +
                                 2 * symbytes_type::value> {};
template <class Variant>
struct ciphertextbytes_type
    : std::integral_constant<unsigned, indcpa_bytes_type<Variant>::value> {};

/* Key exchange params */
template <class Variant>
struct ke_sendabytes_type
    : std::integral_constant<unsigned, publickeybytes_type<Variant>::value> {};
template <class Variant>
struct ke_sendbbytes_type
    : std::integral_constant<unsigned, ciphertextbytes_type<Variant>::value> {};

/* Unilateral authenticated key exchange params */
template <class Variant>
struct uake_sendabytes_type
    : std::integral_constant<unsigned,
                             publickeybytes_type<Variant>::value +
                                 ciphertextbytes_type<Variant>::value> {};
template <class Variant>
struct uake_sendbbytes_type
    : std::integral_constant<unsigned, ciphertextbytes_type<Variant>::value> {};

/* Authenticated key exchange params */
template <class Variant>
struct ake_sendabytes_type
    : std::integral_constant<unsigned,
                             publickeybytes_type<Variant>::value +
                                 ciphertextbytes_type<Variant>::value> {};
template <class Variant>
struct ake_sendbbytes_type
    : std::integral_constant<unsigned,
                             2 * ciphertextbytes_type<Variant>::value> {};

}  // namespace params_type

namespace params {

inline constexpr unsigned n = params_type::n_type::value;
template <class Variant>
inline constexpr unsigned k = params_type::k_type<Variant>::value;
inline constexpr unsigned q = params_type::q_type::value;
template <class Variant>
inline constexpr unsigned eta1 = params_type::eta1_type<Variant>::value;
inline constexpr unsigned eta2 = params_type::eta2_type::value;
template <class Variant>
inline constexpr unsigned du = params_type::du_type<Variant>::value;
template <class Variant>
inline constexpr unsigned dv = params_type::dv_type<Variant>::value;
inline constexpr unsigned symbytes = params_type::symbytes_type::value;
inline constexpr unsigned ssbytes = params_type::ssbytes_type::value;
inline constexpr unsigned polybytes = params_type::polybytes_type::value;
template <class Variant>
inline constexpr unsigned polyvecbytes =
    params_type::polyvecbytes_type<Variant>::value;
template <class Variant>
inline constexpr unsigned polycompressedbytes =
    params_type::polycompressedbytes_type<Variant>::value;
template <class Variant>
inline constexpr unsigned polyveccompressedbytes =
    params_type::polyveccompressedbytes_type<Variant>::value;
inline constexpr unsigned indcpa_msgbytes =
    params_type::indcpa_msgbytes_type::value;
template <class Variant>
inline constexpr unsigned indcpa_publickeybytes =
    params_type::indcpa_publickeybytes_type<Variant>::value;
template <class Variant>
inline constexpr unsigned indcpa_secretkeybytes =
    params_type::indcpa_secretkeybytes_type<Variant>::value;
template <class Variant>
inline constexpr unsigned indcpa_bytes =
    params_type::indcpa_bytes_type<Variant>::value;
template <class Variant>
inline constexpr unsigned publickeybytes =
    params_type::publickeybytes_type<Variant>::value;
template <class Variant>
inline constexpr unsigned secretkeybytes =
    params_type::secretkeybytes_type<Variant>::value;
template <class Variant>
inline constexpr unsigned ciphertextbytes =
    params_type::ciphertextbytes_type<Variant>::value;
template <class Variant>
inline constexpr unsigned ke_sendabytes =
    params_type::ke_sendabytes_type<Variant>::value;
template <class Variant>
inline constexpr unsigned ke_sendbbytes =
    params_type::ke_sendbbytes_type<Variant>::value;
template <class Variant>
inline constexpr unsigned uake_sendabytes =
    params_type::uake_sendabytes_type<Variant>::value;
template <class Variant>
inline constexpr unsigned uake_sendbbytes =
    params_type::uake_sendbbytes_type<Variant>::value;
template <class Variant>
inline constexpr unsigned ake_sendabytes =
    params_type::ake_sendabytes_type<Variant>::value;
template <class Variant>
inline constexpr unsigned ake_sendbbytes =
    params_type::ake_sendbbytes_type<Variant>::value;

}  // namespace params

}  // namespace atpqc_cuda::kyber

#endif
