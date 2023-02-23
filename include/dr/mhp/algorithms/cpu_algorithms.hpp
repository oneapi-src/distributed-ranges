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
  lib::drlog.debug("fill: dr: {}\n", dr);
  for (const auto &s : local_segments(dr)) {
    lib::drlog.debug("fill: segment before: {}\n", s);
    rng::fill(s, value);
    lib::drlog.debug("fill: segment after: {}\n", s);
  }
  barrier();
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

void copy(lib::distributed_contiguous_range auto &&in,
          lib::distributed_iterator auto out) {
  if (aligned(in.begin(), out)) {
    for (const auto &&[in_seg, out_seg] :
         rng::views::zip(local_segments(in), local_segments(out))) {
      rng::copy(in_seg, out_seg.begin());
    }
    barrier();
  } else {
    lib::drlog.debug("copy: serial execution\n");
    rng::copy(in, out);
    fence();
  }
}

template <lib::distributed_iterator DI_IN>
void copy(DI_IN &&first, DI_IN &&last, lib::distributed_iterator auto &&out) {
  mhp::copy(rng::subrange(first, last), out);
}

//
//
// for_each
//
//

/// Collective for_each on distributed range
void for_each(lib::distributed_range auto &&dr, auto op) {
  for (const auto &s : local_segments(dr)) {
    rng::for_each(s, op);
  }
  barrier();
}

/// Collective for_each on iterator/sentinel for a distributed range
template <lib::distributed_iterator DI>
void for_each(DI first, DI last, auto op) {
  mhp::for_each(rng::subrange(first, last), op);
}

//
//
// iota
//
//

/// Collective iota on iterator/sentinel for a distributed range
template <lib::distributed_iterator DI>
void iota(DI first, DI last, auto value) {
  if (default_comm().rank() == 0) {
    std::iota(first, last, value);
  }
  fence();
}

/// Collective iota on distributed range
void iota(lib::distributed_contiguous_range auto &&r, auto value) {
  mhp::iota(r.begin(), r.end(), value);
}

//
//
// Reduce
//
//

/// Collective reduction on a distributed range
template <lib::distributed_iterator DI, typename T>
T reduce(int root, DI first, DI last, T init, auto &&binary_op) {
  T result = 0;
  auto comm = default_comm();

  if (aligned(first)) {
    lib::drlog.debug("Parallel reduce\n");
    // reduce each segment, collect in a vector
    std::vector<T> locals;
    rng::for_each(
        local_segments(rng::subrange(first, last)),
        [&](auto v) { locals.push_back(v); },
        [=](auto sr) {
          return std::reduce(std::execution::par_unseq, sr.begin(), sr.end(),
                             T(0), binary_op);
        });
    // reduce the vector
    auto local = std::reduce(std::execution::par_unseq, locals.begin(),
                             locals.end(), T(0), binary_op);

    // Collect rank values in a vector
    std::vector<T> all(comm.size());
    comm.gather(local, all, root);
    if (comm.rank() == root) {
      result = std::reduce(std::execution::par_unseq, all.begin(), all.end(),
                           init, binary_op);
    }
  } else {
    lib::drlog.debug("Serialreduce\n");
    if (comm.rank() == root) {
      // Reduce on root node
      result =
          std::reduce(std::execution::par_unseq, first, last, init, binary_op);
    }
    barrier();
  }
  return result;
}

//
//
// transform
//
//

void transform(lib::distributed_range auto &&in,
               lib::distributed_iterator auto out, auto op) {
  if (aligned(in.begin(), out)) {
    for (const auto &&[in_seg, out_seg] :
         rng::views::zip(local_segments(in), local_segments(out))) {
      rng::transform(in_seg, out_seg.begin(), op);
    }
    barrier();
  } else {
    lib::drlog.debug("transform: serial execution\n");
    rng::transform(in, out, op);
    fence();
  }
}

template <lib::distributed_iterator DI_IN>
void transform(DI_IN &&first, DI_IN &&last,
               lib::distributed_iterator auto &&out, auto op) {
  mhp::transform(rng::subrange(first, last), out, op);
}

} // namespace mhp
