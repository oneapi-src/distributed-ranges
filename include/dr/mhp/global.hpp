// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

namespace dr::mhp {

namespace __detail {

struct global_context {
  global_context() {}
#ifdef SYCL_LANGUAGE_VERSION
  global_context(sycl::queue q) : sycl_queue_(q), dpl_policy_(q) {}
  sycl::queue sycl_queue_;
  decltype(oneapi::dpl::execution::make_device_policy(
      std::declval<sycl::queue>())) dpl_policy_;
#endif

  bool use_sycl_ = false;
  dr::communicator comm_;
  // container owns the window, we just track MPI handle
  std::set<MPI_Win> wins_;
};

inline global_context *global_context_ = nullptr;

inline auto gcontext() {
  assert(global_context_ && "Call mhp::init() after MPI_Init()");
  return global_context_;
}

} // namespace __detail

inline void final() {
  delete __detail::global_context_;
  __detail::global_context_ = nullptr;
}

inline dr::communicator &default_comm() { return __detail::gcontext()->comm_; }

inline std::set<MPI_Win> &active_wins() { return __detail::gcontext()->wins_; }

inline void barrier() { __detail::gcontext()->comm_.barrier(); }
inline auto use_sycl() { return __detail::gcontext()->use_sycl_; }

inline void fence() {
  dr::drlog.debug("fence\n");
  for (auto win : __detail::gcontext()->wins_) {
    MPI_Win_fence(0, win);
  }
}

inline void init() {
  assert(__detail::global_context_ == nullptr &&
         "Do not call mhp::init() more than once");
  __detail::global_context_ = new __detail::global_context;
}

#ifdef SYCL_LANGUAGE_VERSION
inline auto sycl_queue() { return __detail::gcontext()->sycl_queue_; }
inline auto dpl_policy() { return __detail::gcontext()->dpl_policy_; }

inline void init(sycl::queue q) {
  assert(__detail::global_context_ == nullptr &&
         "Do not call mhp::init() more than once");
  __detail::global_context_ = new __detail::global_context(q);
}
#else
inline auto sycl_queue() {
  assert(false);
  return 0;
}
inline const auto &dpl_policy() {
  assert(false);
  return std::execution::seq;
}
#endif
} // namespace dr::mhp
