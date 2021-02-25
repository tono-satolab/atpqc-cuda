//
// device.cu
// Device function of absorb, squeeze,
// and permutation for Keccak.
//
// Copyright (c) 2021 Tatsuki Ono
//
// This software is released under the MIT License.
// https://opensource.org/licenses/mit-license.php
//

#include "device.cuh"

namespace atpqc_cuda::fips202_ws::device {

namespace {

constexpr unsigned nrounds = 24;

__device__ const std::uint64_t keccak_f_round_constants[nrounds] = {
    0x0000'0000'0000'0001ULL, 0x0000'0000'0000'8082ULL,
    0x8000'0000'0000'808aULL, 0x8000'0000'8000'8000ULL,
    0x0000'0000'0000'808bULL, 0x0000'0000'8000'0001ULL,
    0x8000'0000'8000'8081ULL, 0x8000'0000'0000'8009ULL,
    0x0000'0000'0000'008aULL, 0x0000'0000'0000'0088ULL,
    0x0000'0000'8000'8009ULL, 0x0000'0000'8000'000aULL,
    0x0000'0000'8000'808bULL, 0x8000'0000'0000'008bULL,
    0x8000'0000'0000'8089ULL, 0x8000'0000'0000'8003ULL,
    0x8000'0000'0000'8002ULL, 0x8000'0000'0000'0080ULL,
    0x0000'0000'0000'800aULL, 0x8000'0000'8000'000aULL,
    0x8000'0000'8000'8081ULL, 0x8000'0000'0000'8080ULL,
    0x0000'0000'8000'0001ULL, 0x8000'0000'8000'8008ULL};

}  // namespace

__device__ const unsigned offset_constants[32] = {
    /* x = 0, 1, 2, 3, 4 */
    0,  1,  62, 28, 27,  // y = 0
    36, 44, 6,  55, 20,  // y = 1
    3,  10, 43, 25, 39,  // y = 2
    41, 45, 15, 21, 8,   // y = 3
    18, 2,  61, 56, 14   // y = 4
};

[[nodiscard]] __device__ keccak::state_type keccak::f1600_state_permute(
    state_type state) const noexcept {
  auto rol = [](state_type a, unsigned b, unsigned c) noexcept -> state_type {
    return (a << b) ^ (a >> c);
  };

  if (threadIdx.x < 25) {
    const unsigned active_mask = __activemask();
    const unsigned x = params0.x;
    // const unsigned y = params0.y;
    const unsigned theta1 = params0.z;
    const unsigned theta4 = params0.w;
    const unsigned offset = params1.x;
    const unsigned rp = params1.y;
    const unsigned chi1 = params1.z;
    const unsigned chi2 = params1.w;

    std::uint64_t a = state;
    std::uint64_t c;

    for (unsigned round = 0; round < nrounds; ++round) {
      // theta
      c = __shfl_sync(active_mask, a, x + 0) ^
          __shfl_sync(active_mask, a, x + 5) ^
          __shfl_sync(active_mask, a, x + 10) ^
          __shfl_sync(active_mask, a, x + 15) ^
          __shfl_sync(active_mask, a, x + 20);
      a = a ^ (__shfl_sync(active_mask, c, theta4) ^
               rol(__shfl_sync(active_mask, c, theta1), 1, 63));

      // rho and pi
      c = __shfl_sync(active_mask, rol(a, offset, 64 - offset), rp);

      // chi
      a = c ^ ((~__shfl_sync(active_mask, c, chi1)) &
               __shfl_sync(active_mask, c, chi2));

      // iota
      if (threadIdx.x == 0) a = a ^ __ldg(&keccak_f_round_constants[round]);
    }

    state = a;
  }

  return state;
}

[[nodiscard]] __device__ keccak::state_type keccak::absorb(
    unsigned rate, const std::uint8_t* m, std::size_t mlen,
    std::uint8_t p) const {
  auto load64 = [](const std::uint8_t x[8]) -> state_type {
    state_type r = static_cast<state_type>(x[0]) << 0;
    r |= static_cast<state_type>(x[1]) << 8;
    r |= static_cast<state_type>(x[2]) << 16;
    r |= static_cast<state_type>(x[3]) << 24;
    r |= static_cast<state_type>(x[4]) << 32;
    r |= static_cast<state_type>(x[5]) << 40;
    r |= static_cast<state_type>(x[6]) << 48;
    r |= static_cast<state_type>(x[7]) << 56;
    return r;
  };

  m += 8 * threadIdx.x;

  state_type state = 0;
  while (mlen >= rate) {
    if (threadIdx.x < rate / 8) {
      state ^= load64(m);
    }
    state = f1600_state_permute(state);
    mlen -= rate;
    m += rate;
  }

  for (unsigned i = 0; i < 8; ++i) {
    state ^= (8 * threadIdx.x + i < mlen)
                 ? static_cast<std::uint64_t>(m[i]) << (8 * i)
                 : 0;
  }
  if (mlen / 8 == threadIdx.x) {
    state ^= static_cast<std::uint64_t>(p) << ((mlen % 8) * 8);
  }
  if ((rate - 1) / 8 == threadIdx.x) {
    state ^= 0x8000'0000'0000'0000ULL;
  }

  return state;
}

[[nodiscard]] __device__ keccak::state_type keccak::squeezeblocks(
    std::uint8_t* out, std::size_t nblocks, state_type state,
    unsigned rate) const {
  auto store64 = [](std::uint8_t x[8], state_type s) {
    x[0] = s >> 0;
    x[1] = s >> 8;
    x[2] = s >> 16;
    x[3] = s >> 24;
    x[4] = s >> 32;
    x[5] = s >> 40;
    x[6] = s >> 48;
    x[7] = s >> 56;
  };

  while (nblocks > 0) {
    state = f1600_state_permute(state);
    if (threadIdx.x < rate / 8) {
      store64(&out[8 * threadIdx.x], state);
    }
    out += rate;
    --nblocks;
  }

  return state;
}

__device__ keccak::keccak() {
  unsigned x = threadIdx.x % 5;
  unsigned y = threadIdx.x / 5;
  unsigned theta1 = (x + 1) % 5;  // (x + 1) mod 5
  unsigned theta4 = (x + 4) % 5;  // (x - 1) mod 5
  unsigned offset = offset_constants[threadIdx.x];
  unsigned rp = (x + 3 * y) % 5 + 5 * x;
  unsigned chi1 = threadIdx.x - x + theta1;       // 5y + (x + 1) % 5
  unsigned chi2 = threadIdx.x - x + (x + 2) % 5;  // 5y + (x + 2) % 5

  params0 = make_char4(x, y, theta1, theta4);
  params1 = make_char4(offset, rp, chi1, chi2);
}

}  // namespace atpqc_cuda::fips202_ws::device
