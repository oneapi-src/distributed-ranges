// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

namespace mhp {

//
//
// fill
//
//

/// Collective fill on distributed range
void fill(lib::distributed_contiguous_range auto &&dr, auto value) {
  for (const auto &s : local_segments(lib::ranges::segments(dr))) {
    rng::fill(s, value);
  }
  dr.begin().barrier();
}

/// Collective fill on iterator/sentinel for a distributed range
template <lib::distributed_iterator DI>
void fill(DI first, DI last, auto value) {
  mhp::fill(rng::subrange(first, last), value);
}

//
//
// for_each
//
//

/// Collective for_each on iterator/sentinel for a distributed range
template <lib::distributed_iterator DI>
void for_each(DI first, DI last, auto op) {
  mhp::for_each(rng::subrange(first, last), op);
}

/// Collective for_each on distributed range
void for_each(lib::distributed_contiguous_range auto &&dr, auto op) {
  for (const auto &s : local_segments(lib::ranges::segments(dr))) {
    rng::for_each(s, op);
  }
  dr.begin().barrier();
}

//
//
// iota
//
//

/// Collective iota on iterator/sentinel for a distributed range
template <lib::distributed_iterator DI>
void iota(DI first, DI last, auto value) {
  if (first.my_rank() == 0) {
    std::iota(first, last, value);
  }
  first.fence();
}

/// Collective iota on distributed range
void iota(lib::distributed_contiguous_range auto &&r, auto value) {
  mhp::iota(r.begin(), r.end(), value);
}

} // namespace mhp
