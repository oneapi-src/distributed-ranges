// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

namespace mhp {

namespace _details {

struct global_context {
  lib::communicator comm_;
  // container owns the window, we just track MPI handle
  std::set<MPI_Win> wins_;
};

inline global_context *global_context_ = nullptr;

inline auto gcontext() {
  assert(global_context_ && "Call mhp::init() after MPI_Init()");
  return global_context_;
}

} // namespace _details

inline void init() {
  assert(_details::global_context_ == nullptr &&
         "Do not call mhp::init() more than once");
  _details::global_context_ = new _details::global_context;
}

inline void final() {
  delete _details::global_context_;
  _details::global_context_ = nullptr;
}

inline lib::communicator &default_comm() { return _details::gcontext()->comm_; }

inline std::set<MPI_Win> &active_wins() { return _details::gcontext()->wins_; }

inline void barrier() { _details::gcontext()->comm_.barrier(); }

inline void fence() {
  lib::drlog.debug("fence\n");
  for (auto win : _details::gcontext()->wins_) {
    lib::drlog.debug("  win: {}\n", win);
    MPI_Win_fence(0, win);
  }
}

} // namespace mhp
