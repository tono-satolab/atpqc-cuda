//
// wrapper.cu
// Wrapper of RNG.
//
// Copyright (c) 2021 Tatsuki Ono
//
// This software is released under the MIT License.
// https://opensource.org/licenses/mit-license.php
//

#ifndef ATPQC_CUDA_LIB_RNG_WRAPPER_HPP_
#define ATPQC_CUDA_LIB_RNG_WRAPPER_HPP_

#include <cstddef>
#include <cstdint>
#include <utility>

namespace atpqc_cuda::rng {

template <class Rng>
class cuda_hostfunc_wrapper {
 private:
  using rng_type = Rng;

 public:
  using args_type = std::pair<std::uint8_t*, std::size_t>;

  static void randombytes(void* args_ptr) {
    rng_type rng;
    const args_type args_pair = *static_cast<args_type*>(args_ptr);
    rng(args_pair.first, args_pair.second);
  }
};

}  // namespace atpqc_cuda::rng

#endif
