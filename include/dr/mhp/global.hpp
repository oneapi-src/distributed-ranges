// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

namespace mhp {

namespace __detail {

struct global_context {
  lib::communicator comm_;
  // container owns the window, we just track MPI handle
  std::set<MPI_Win> wins_;
#ifdef SYCL_LANGUAGE_VERSION
  sycl::queue sycl_queue_;
#endif
};

inline global_context *global_context_ = nullptr;

inline auto gcontext() {
  assert(global_context_ && "Call mhp::init() after MPI_Init()");
  return global_context_;
}

} // namespace __detail

inline void init() {
  assert(__detail::global_context_ == nullptr &&
         "Do not call mhp::init() more than once");
  __detail::global_context_ = new __detail::global_context;
}

inline void final() {
  delete __detail::global_context_;
  __detail::global_context_ = nullptr;
}

inline lib::communicator &default_comm() { return __detail::gcontext()->comm_; }

inline std::set<MPI_Win> &active_wins() { return __detail::gcontext()->wins_; }

inline void barrier() { __detail::gcontext()->comm_.barrier(); }

inline void fence() {
  lib::drlog.debug("fence\n");
  for (auto win : __detail::gcontext()->wins_) {
    lib::drlog.debug("  win: {}\n", win);
    MPI_Win_fence(0, win);
  }
}

#ifdef SYCL_LANGUAGE_VERSION
inline auto sycl_queue() { return __detail::gcontext()->sycl_queue_; }

inline void init(sycl::queue q) {
  init();
  __detail::gcontext()->sycl_queue_ = q;
}
#endif

} // namespace mhp
