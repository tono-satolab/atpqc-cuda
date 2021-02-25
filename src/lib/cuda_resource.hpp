//
// cuda_debug.hpp
// Classes for resources of CUDA.
//
// Copyright (c) 2021 Tatsuki Ono
//
// This software is released under the MIT License.
// https://opensource.org/licenses/mit-license.php
//

#ifndef ATPQC_CUDA_LIB_CUDA_RESOURCE_HPP_
#define ATPQC_CUDA_LIB_CUDA_RESOURCE_HPP_

#include <cuda.h>

#include <tuple>

#include "cuda_debug.hpp"

namespace atpqc_cuda::cuda_resource {

class context {
 private:
  CUcontext ctx;

 public:
  context() = delete;
  explicit context(CUdevice dev) : ctx(nullptr) {
    CCC(cuCtxCreate(&ctx, CU_CTX_SCHED_AUTO, dev));
  }
  explicit context(CUdevice dev, unsigned flags) : ctx(nullptr) {
    CCC(cuCtxCreate(&ctx, flags, dev));
  }
  ~context() {
    if (ctx != nullptr) CCC(cuCtxDestroy(ctx));
  }
  context(const context&) = delete;
  context& operator=(const context&) = delete;
  context(context&&) = delete;
  context& operator=(context&&) = delete;
  operator CUcontext() const noexcept { return ctx; }
};

template <class T>
class device_memory {
 private:
  T* ptr;

 public:
  device_memory() = delete;
  explicit device_memory(std::size_t size) : ptr(nullptr) {
    CCC(cudaMalloc(&ptr, sizeof(T) * size));
  }
  ~device_memory() {
    if (ptr != nullptr) CCC(cudaFree(ptr));
  }
  device_memory(const device_memory&) = delete;
  device_memory& operator=(const device_memory&) = delete;
  device_memory(device_memory&& r) noexcept : ptr(r.ptr) { r.ptr = nullptr; }
  device_memory& operator=(device_memory&& r) & {
    if (ptr != nullptr) CCC(cudaFree(ptr));
    ptr = r.ptr;
    r.ptr = nullptr;
    return *this;
  }
  T* get_ptr() const noexcept { return ptr; }
};

template <class T>
class device_pitched_memory {
 private:
  T* ptr;
  std::size_t pitch;

 public:
  device_pitched_memory() = delete;
  explicit device_pitched_memory(std::size_t width, std::size_t height)
      : ptr(nullptr) {
    CCC(cudaMallocPitch(&ptr, &pitch, sizeof(T) * width, height));
  }
  ~device_pitched_memory() {
    if (ptr != nullptr) CCC(cudaFree(ptr));
  }
  device_pitched_memory(const device_pitched_memory&) = delete;
  device_pitched_memory& operator=(const device_pitched_memory&) = delete;
  device_pitched_memory(device_pitched_memory&& r) noexcept
      : ptr(r.ptr), pitch(r.pitch) {
    r.ptr = nullptr;
  }
  device_pitched_memory& operator=(device_pitched_memory&& r) & {
    if (ptr != nullptr) CCC(cudaFree(ptr));
    ptr = r.ptr;
    pitch = r.pitch;
    r.ptr = nullptr;
    return *this;
  }
  T* get_ptr() const noexcept { return ptr; }
  std::size_t get_pitch() const noexcept { return pitch; }
};

template <class T>
class pinned_memory {
 private:
  T* ptr;

 public:
  pinned_memory() = delete;
  explicit pinned_memory(std::size_t size) : ptr(nullptr) {
    CCC(cudaMallocHost(&ptr, sizeof(T) * size));
  }
  explicit pinned_memory(std::size_t size, unsigned flags) : ptr(nullptr) {
    CCC(cudaHostAlloc(&ptr, sizeof(T) * size, flags));
  }
  ~pinned_memory() {
    if (ptr != nullptr) CCC(cudaFreeHost(ptr));
  }
  pinned_memory(const pinned_memory&) = delete;
  pinned_memory& operator=(const pinned_memory&) = delete;
  pinned_memory(pinned_memory&& r) noexcept : ptr(r.ptr) { r.ptr = nullptr; }
  pinned_memory& operator=(pinned_memory&& r) & {
    if (ptr != nullptr) CCC(cudaFreeHost(ptr));
    ptr = r.ptr;
    r.ptr = nullptr;
    return *this;
  }
  T* get_ptr() const noexcept { return ptr; }
};

class stream {
 private:
  cudaStream_t s;

 public:
  explicit stream() : s(nullptr) { CCC(cudaStreamCreate(&s)); }
  explicit stream(unsigned flags) : s(nullptr) {
    CCC(cudaStreamCreateWithFlags(&s, flags));
  }
  explicit stream(unsigned flags, int priority) : s(nullptr) {
    CCC(cudaStreamCreateWithPriority(&s, flags, priority));
  }
  ~stream() {
    if (s != nullptr) CCC(cudaStreamDestroy(s));
  }
  stream(const stream&) = delete;
  stream& operator=(const stream&) = delete;
  stream(stream&& r) noexcept : s(r.s) { r.s = nullptr; }
  stream& operator=(stream&& r) & {
    if (s != nullptr) CCC(cudaStreamDestroy(s));
    s = r.s;
    r.s = nullptr;
    return *this;
  }
  operator cudaStream_t() const noexcept { return s; }
};

class event {
 private:
  cudaEvent_t e;

 public:
  explicit event() : e(nullptr) { CCC(cudaEventCreate(&e)); }
  explicit event(unsigned flags) : e(nullptr) {
    CCC(cudaEventCreateWithFlags(&e, flags));
  }
  ~event() {
    if (e != nullptr) CCC(cudaEventDestroy(e));
  }
  event(const event&) = delete;
  event& operator=(const event&) = delete;
  event(event&& r) noexcept : e(r.e) { r.e = nullptr; }
  event& operator=(event&& r) & {
    if (e != nullptr) CCC(cudaEventDestroy(e));
    e = r.e;
    r.e = nullptr;
    return *this;
  }
  operator cudaEvent_t() const noexcept { return e; }
};

class graph {
 private:
  cudaGraph_t g;

 public:
  graph() : g(nullptr) { CCC(cudaGraphCreate(&g, 0)); }
  ~graph() {
    if (g != nullptr) CCC(cudaGraphDestroy(g));
  }
  graph(const graph& r) : g(nullptr) { CCC(cudaGraphClone(&g, r.g)); }
  graph& operator=(const graph& r) & {
    if (g != nullptr) CCC(cudaGraphDestroy(g));
    CCC(cudaGraphClone(&g, r.g));
    return *this;
  }
  graph(graph&& r) noexcept : g(r.g) { r.g = nullptr; }
  graph& operator=(graph&& r) & {
    if (g != nullptr) CCC(cudaGraphDestroy(g));
    g = r.g;
    r.g = nullptr;
    return *this;
  }
  operator cudaGraph_t() const noexcept { return g; }
};

class graph_exec {
 private:
  cudaGraphExec_t ge;

 public:
  graph_exec() = delete;
  explicit graph_exec(cudaGraph_t g) : ge(nullptr) {
    CCC(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0));
  }
  explicit graph_exec(cudaGraph_t g, cudaGraphNode_t* error_node_ptr,
                      char* log_buf, std::size_t buf_size)
      : ge(nullptr) {
    CCC(cudaGraphInstantiate(&ge, g, error_node_ptr, log_buf, buf_size));
  }
  ~graph_exec() {
    if (ge != nullptr) CCC(cudaGraphExecDestroy(ge));
  }
  graph_exec(const graph_exec&) = delete;
  graph_exec& operator=(const graph_exec&) = delete;
  graph_exec(graph_exec&& r) noexcept : ge(r.ge) { r.ge = nullptr; }
  graph_exec& operator=(graph_exec&& r) & {
    if (ge != nullptr) CCC(cudaGraphExecDestroy(ge));
    ge = r.ge;
    r.ge = nullptr;
    return *this;
  }
  operator cudaGraphExec_t() const noexcept { return ge; }
};

template <typename... Args>
class kernel_args {
 public:
  using args_value_type = std::tuple<Args...>;
  using args_num_type = std::tuple_size<args_value_type>;
  using args_ptr_type = std::array<void*, args_num_type::value>;

 private:
  args_value_type val;
  args_ptr_type ptr;

  template <std::size_t N = 0>
  void make_ptr_array() noexcept {
    if constexpr (N < args_num_type::value) {
      ptr[N] =
          const_cast<void*>(reinterpret_cast<const void*>(&std::get<N>(val)));
      make_ptr_array<N + 1>();
    }
  }

 public:
  kernel_args() = delete;
  kernel_args(Args... args) noexcept : val(args...) { make_ptr_array(); }
  kernel_args(const kernel_args& r) noexcept : val(r.val) { make_ptr_array(); };
  kernel_args& operator=(const kernel_args& r) & noexcept { val = r.val; };
  kernel_args(kernel_args&& r) noexcept : val(r.val) { make_ptr_array(); }
  kernel_args& operator=(kernel_args&& r) & noexcept { val = std::move(r.val); }

  void** get_args_ptr() const noexcept {
    return const_cast<void**>(ptr.data());
  }
};

}  // namespace atpqc_cuda::cuda_resource

#endif
