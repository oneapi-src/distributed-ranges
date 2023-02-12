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
  for (const auto &s : local_segments(dr)) {
    rng::fill(s, value);
  }
  mhp::barrier(dr.begin());
}

/// Collective fill on iterator/sentinel for a distributed range
template <lib::distributed_iterator DI>
void fill(DI first, DI last, auto value) {
  mhp::fill(rng::subrange(first, last), value);
}

//
//
// copy
//
//

template <lib::distributed_contiguous_range DR_IN, typename DI_OUT>
void copy(DR_IN &&in, DI_OUT out) {
  if (conformant(in.begin(), out)) {
    for (const auto &&[in_seg, out_seg] :
         rng::views::zip(local_segments(in), local_segments(out))) {
      rng::copy(in_seg, out_seg.begin());
    }
    mhp::barrier(out);
  } else {
    rng::copy(in, out);
    mhp::fence(out);
  }
}

template <lib::distributed_iterator DI_IN, lib::distributed_iterator DI_OUT>
void copy(DI_IN &&first, DI_IN &&last, DI_OUT &&out) {
  mhp::copy(rng::subrange(first, last), out);
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
void for_each(lib::distributed_range auto &&dr, auto op) {
  for (const auto &s : local_segments(dr)) {
    rng::for_each(s, op);
  }
  mhp::barrier(dr.begin());
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
  mhp::fence(first);
}

/// Collective iota on distributed range
void iota(lib::distributed_contiguous_range auto &&r, auto value) {
  mhp::iota(r.begin(), r.end(), value);
}

//
//
// transform
//
//

template <lib::distributed_contiguous_range DR_IN, typename DI_OUT>
void transform(DR_IN &&in, DI_OUT out, auto op) {
  if (conformant(in.begin(), out)) {
    for (const auto &&[in_seg, out_seg] :
         rng::views::zip(local_segments(in), local_segments(out))) {
      rng::transform(in_seg, out_seg.begin(), op);
    }
    mhp::barrier(out);
  } else {
    rng::transform(in, out, op);
    mhp::fence(out);
  }
}

template <lib::distributed_iterator DI_IN, lib::distributed_iterator DI_OUT>
void transform(DI_IN &&first, DI_IN &&last, DI_OUT &&out, auto op) {
  mhp::transform(rng::subrange(first, last), out, op);
}

} // namespace mhp
