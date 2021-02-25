//
// device.cuh
// Device function of SHA-3.
//
// Copyright (c) 2021 Tatsuki Ono
//
// This software is released under the MIT License.
// https://opensource.org/licenses/mit-license.php
//

#ifndef ATPQC_CUDA_LIB_FIPS202_WS_DEVICE_CUH_
#define ATPQC_CUDA_LIB_FIPS202_WS_DEVICE_CUH_

#include <cstddef>
#include <cstdint>

#include "params.cuh"

namespace atpqc_cuda::fips202_ws::device {

__device__ extern const unsigned offset_constants[32];

class keccak {
 public:
  using state_type = std::uint64_t;

 private:
  char4 params0;
  char4 params1;

  [[nodiscard]] __device__ state_type
  f1600_state_permute(state_type state) const noexcept;

 public:
  [[nodiscard]] __device__ state_type absorb(unsigned rate,
                                             const std::uint8_t* m,
                                             std::size_t mlen,
                                             std::uint8_t p) const;
  [[nodiscard]] __device__ state_type squeezeblocks(std::uint8_t* out,
                                                    std::size_t nblocks,
                                                    state_type state,
                                                    unsigned rate) const;
  __device__ keccak();
};

template <unsigned Bits>
class shake {
 public:
  using keccak_type = keccak;
  static constexpr unsigned rate = params::shake<Bits>::rate;

 private:
  using state_type = keccak_type::state_type;

 public:
  class absorb {
   public:
    using keccak_type = keccak;
    using state_type = keccak_type::state_type;
    static constexpr unsigned rate = params::shake<Bits>::rate;

    [[nodiscard]] __device__ state_type operator()(const std::uint8_t* in,
                                                   std::size_t inlen,
                                                   const keccak_type& f) const {
      return f.absorb(rate, in, inlen, 0x1f);
    }
  };

  class squeezeblocks {
   public:
    using keccak_type = keccak;
    using state_type = keccak_type::state_type;
    static constexpr unsigned rate = params::shake<Bits>::rate;

    [[nodiscard]] __device__ state_type operator()(std::uint8_t* out,
                                                   std::size_t nblocks,
                                                   state_type state,
                                                   const keccak_type& f) const {
      return f.squeezeblocks(out, nblocks, state, rate);
    }
  };

  __device__ void operator()(std::uint8_t* out, std::size_t outlen,
                             const std::uint8_t* in, std::size_t inlen,
                             const keccak_type& f,
                             std::uint8_t* tmp_shared) const {
    absorb absorb_f;
    squeezeblocks squeezeblocks_f;

    const std::size_t nblocks = outlen / rate;
    const std::size_t blocks_len = nblocks * rate;

    state_type state = absorb_f(in, inlen, f);
    state = squeezeblocks_f(out, nblocks, state, f);

    outlen -= blocks_len;

    if (outlen > 0) {
      out += blocks_len;
      state = squeezeblocks_f(tmp_shared, 1, state, f);

      __syncwarp();

      for (unsigned i = threadIdx.x; i < outlen; i += 32) {
        out[i] = tmp_shared[i];
      }
    }
  }
};

template <unsigned Bits>
class notmp_shake {
 public:
  using keccak_type = keccak;
  static constexpr unsigned rate = params::shake<Bits>::rate;

 private:
  using absorb_type = typename shake<Bits>::absorb;
  using squeezeblocks_type = typename shake<Bits>::squeezeblocks;
  using state_type = keccak_type::state_type;

 public:
  __device__ __host__ static constexpr unsigned len_to_nblocks(
      unsigned len) noexcept {
    return (len + (rate - 1)) / rate;
  }

  __device__ void operator()(std::uint8_t* out_blocksized,
                             std::size_t outblocks, const std::uint8_t* in,
                             std::size_t inlen, const keccak_type& f) const {
    absorb_type absorb;
    squeezeblocks_type squeezeblocks;

    state_type state = absorb(in, inlen, f);
    static_cast<void>(squeezeblocks(out_blocksized, outblocks, state, f));
  }
};

template <unsigned Bits>
class sha3 {
 public:
  using keccak_type = keccak;
  static constexpr unsigned rate = params::sha3<Bits>::rate;
  static constexpr unsigned outputbytes = params::sha3<Bits>::outputbytes;

 private:
  using state_type = keccak_type::state_type;

 public:
  __device__ void operator()(std::uint8_t* h, const std::uint8_t* in,
                             std::size_t inlen, const keccak_type& f,
                             std::uint8_t* tmp_shared) const {
    state_type state = f.absorb(rate, in, inlen, 0x06);
    static_cast<void>(f.squeezeblocks(tmp_shared, 1, state, rate));

    __syncwarp();

    for (unsigned i = threadIdx.x; i < outputbytes; i += 32) {
      h[i] = tmp_shared[i];
    }
  }
};

template <unsigned Bits>
class notmp_sha3 {
 public:
  using keccak_type = keccak;
  static constexpr unsigned rate = params::sha3<Bits>::rate;
  static constexpr unsigned outputbytes = params::sha3<Bits>::outputbytes;

 private:
  using state_type = keccak_type::state_type;

 public:
  __device__ void operator()(std::uint8_t* h_ratesized, const std::uint8_t* in,
                             const std::size_t inlen,
                             const keccak_type& f) const {
    state_type state = f.absorb(rate, in, inlen, 0x06);
    static_cast<void>(f.squeezeblocks(h_ratesized, 1, state, rate));
  }
};

}  // namespace atpqc_cuda::fips202_ws::device

#endif
