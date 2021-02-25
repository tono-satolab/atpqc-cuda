//
// cuda_debug.hpp
// Macros for debugging CUDA API call.
//
// Copyright (c) 2021 Tatsuki Ono
//
// This software is released under the MIT License.
// https://opensource.org/licenses/mit-license.php
//

#ifndef ATPQC_CUDA_LIB_CUDA_DEBUG_HPP_
#define ATPQC_CUDA_LIB_CUDA_DEBUG_HPP_

#ifdef CUDA_DEBUG

#include <cuda.h>

#include <cstdio>

#define CCE()                                                           \
  cuda_debug::check_cuda_call(cudaDeviceSynchronize(), "CCE", __FILE__, \
                              __LINE__)
#define CCC(call) cuda_debug::check_cuda_call(call, #call, __FILE__, __LINE__)
#define CUDA_DEBUG_RESET() cuda_debug::reset()

class cuda_debug {
 private:
  static inline bool error_occured;

 public:
  static CUresult check_cuda_call(CUresult result, const char* const func,
                                  const char* const file, const unsigned line) {
    if (error_occured == false && result != CUDA_SUCCESS) {
      const char* err_name = nullptr;
      const char* err_str = nullptr;
      cuGetErrorName(result, &err_name);
      cuGetErrorString(result, &err_str);
      std::fprintf(stderr, "CUDA error at %s:%d\n", file, line);
      std::fprintf(stderr, "function=\"%s\"\n", func);
      std::fprintf(stderr, "code=%d \"%s\"\n", static_cast<unsigned>(result),
                   err_name);
      std::fprintf(stderr, "%s\n", err_str);

      error_occured = true;
    }
    return result;
  }

  static cudaError_t check_cuda_call(cudaError_t result, const char* const func,
                                     const char* const file,
                                     const unsigned line) {
    if (error_occured == false && result != cudaSuccess) {
      const char* err_name = cudaGetErrorName(result);
      const char* err_str = cudaGetErrorString(result);
      std::fprintf(stderr, "CUDA error at %s:%d\n", file, line);
      std::fprintf(stderr, "function=\"%s\"\n", func);
      std::fprintf(stderr, "code=%d \"%s\"\n", static_cast<unsigned>(result),
                   err_name);
      std::fprintf(stderr, "%s\n", err_str);

      error_occured = true;
    }
    return result;
  }

  static void reset() { error_occured = false; }
};

#else

#define CCE()
#define CCC(call) call
#define CUDA_DEBUG_RESET()

#endif

#endif
