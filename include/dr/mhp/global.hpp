// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

namespace mhp {

class global_context;

inline global_context *global_context_;

class global_context {
private:
  friend void barrier();
  friend void fence();
  friend lib::communicator &default_comm();
  friend std::set<MPI_Win> &active_wins();

  lib::communicator comm_;
  // container owns the window, we just track MPI handle
  std::set<MPI_Win> wins_;
};

inline void init() { global_context_ = new global_context; }

inline void final() { delete global_context_; }

inline lib::communicator &default_comm() { return global_context_->comm_; }

inline std::set<MPI_Win> &active_wins() { return global_context_->wins_; }

inline void barrier() { global_context_->comm_.barrier(); }

inline void fence() {
  lib::drlog.debug("fence\n");
  for (auto win : global_context_->wins_) {
    lib::drlog.debug("  win: {}\n", win);
    MPI_Win_fence(0, win);
  }
}

} // namespace mhp
