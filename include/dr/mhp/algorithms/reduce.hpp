// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

namespace dr::mhp::__detail {

inline auto std_reduce(rng::forward_range auto &&r, auto &&binary_op) {
  using value_type = rng::range_value_t<decltype(r)>;
  if (rng::empty(r)) {
    return value_type{};
  } else {
    auto skip1 = rng::begin(r);
    skip1++;
    // Explicit cast from distributed_vector reference to value_type
    return std::reduce(std::execution::par_unseq, skip1, rng::end(r),
                       value_type(*rng::begin(r)), binary_op);
  }
}

inline auto dpl_reduce(rng::forward_range auto &&r, auto &&binary_op) {
  rng::range_value_t<decltype(r)> none{};
  if (rng::empty(r)) {
    return none;
  } else {
    return std::reduce(
        dpl_policy(), dr::__detail::direct_iterator(rng::begin(r) + 1),
        dr::__detail::direct_iterator(rng::end(r)), *rng::begin(r), binary_op);
  }
}

/// handles everything but init
template <dr::distributed_range DR>
auto reduce(std::size_t root, bool root_provided, DR &&dr, auto &&binary_op) {
  using value_type = rng::range_value_t<DR>;
  auto comm = default_comm();

  if (rng::empty(dr)) {
    return rng::range_value_t<DR>{};
  }

  if (aligned(dr)) {
    dr::drlog.debug("Parallel reduce\n");

    // Reduce the local segments
    auto reduce = [=](auto &&r) {
      assert(rng::size(r) > 0);
      if (mhp::use_sycl()) {
        dr::drlog.debug("  with DPL\n");
        return dpl_reduce(r, binary_op);
      } else {
        dr::drlog.debug("  with CPU\n");
        return std_reduce(r, binary_op);
      }
    };
    auto locals = rng::views::transform(local_segments(dr), reduce);
    auto local = std_reduce(locals, binary_op);

    std::vector<value_type> all(comm.size()); // dr-style ignore
    if (root_provided) {
      // Everyone gathers to root, only root reduces
      comm.gather(local, std::span{all}, root);
      if (root == comm.rank()) {
        return std_reduce(all, binary_op);
      } else {
        return value_type{};
      }
    } else {
      // Everyone gathers and everyone reduces
      comm.all_gather(local, all);
      return std_reduce(all, binary_op);
    }
  } else {
    dr::drlog.debug("Serial reduce\n");
    value_type result{};
    if (!root_provided || root == comm.rank()) {
      result = std_reduce(dr, binary_op);
    }
    barrier();
    return result;
  }
}

// handles init
template <typename T, dr::distributed_range DR>
T reduce(std::size_t root, bool root_provided, DR &&dr, T init,
         auto &&binary_op = std::plus<>{}) {
  return binary_op(init, reduce(root, root_provided, dr, binary_op));
}

inline void
#if defined(__GNUC__) && !defined(__clang__)
    __attribute__((optimize(0)))
#endif
    no_optimize(auto x) {
}

}; // namespace dr::mhp::__detail

namespace dr::mhp {

//
// Ranges
//

// range, init, and binary op, w/wo root

/// Collective reduction on a distributed range
template <typename T, dr::distributed_range DR>
auto reduce(std::size_t root, DR &&dr, T init, auto &&binary_op) {
  return __detail::reduce(root, true, std::forward<DR>(dr), init, binary_op);
}
/// Collective reduction on a distributed range
template <typename T, dr::distributed_range DR>
auto reduce(DR &&dr, T init, auto &&binary_op) {
  return __detail::reduce(0, false, std::forward<DR>(dr), init, binary_op);
}

// range, init, w/wo root

/// Collective reduction on a distributed range
template <typename T, dr::distributed_range DR>
auto reduce(std::size_t root, DR &&dr, T init) {
  return __detail::reduce(root, true, std::forward<DR>(dr), init,
                          std::plus<>{});
}
/// Collective reduction on a distributed range
template <typename T, dr::distributed_range DR> auto reduce(DR &&dr, T init) {
  return __detail::reduce(0, false, std::forward<DR>(dr), init, std::plus<>{});
}

// range, w/wo root

/// Collective reduction on a distributed range
template <dr::distributed_range DR> auto reduce(std::size_t root, DR &&dr) {
  return __detail::reduce(root, true, std::forward<DR>(dr), std::plus<>{});
}

/// Collective reduction on a distributed range
template <dr::distributed_range DR> auto reduce(DR &&dr) {
  auto x = __detail::reduce(0, false, std::forward<DR>(dr), std::plus<>{});

  // The code below avoids an issue where DotProduct_ZipReduce_DR
  // fails with gcc11.  From debugging, I can see that the call to
  // __detail::reduce above computes the correct value, but this
  // function returns a bad value. My theory is that the problem is
  // related to tail call optimization and the function below disables
  // the optimization.
  __detail::no_optimize(x);

  return x;
}

//
// Iterators
//

// range, init, and binary op, w/wo root

/// Collective reduction on a distributed range
template <typename T, dr::distributed_iterator DI>
auto reduce(std::size_t root, DI first, DI last, T init, auto &&binary_op) {
  return __detail::reduce(root, true, rng::subrange(first, last), init,
                          binary_op);
}
/// Collective reduction on a distributed range
template <typename T, dr::distributed_iterator DI>
auto reduce(DI first, DI last, T init, auto &&binary_op) {
  return __detail::reduce(0, false, rng::subrange(first, last), init,
                          binary_op);
}

// range, init, w/wo root

/// Collective reduction on a distributed range
template <typename T, dr::distributed_iterator DI>
auto reduce(std::size_t root, DI first, DI last, T init) {
  return __detail::reduce(root, true, rng::subrange(first, last), init,
                          std::plus<>{});
}
/// Collective reduction on a distributed range
template <typename T, dr::distributed_iterator DI>
auto reduce(DI first, DI last, T init) {
  return __detail::reduce(0, false, rng::subrange(first, last), init,
                          std::plus<>{});
}

// range, w/wo root

/// Collective reduction on a distributed range
template <dr::distributed_iterator DI>
auto reduce(std::size_t root, DI first, DI last) {
  return __detail::reduce(root, true, rng::subrange(first, last),
                          std::plus<>{});
}
/// Collective reduction on a distributed range
template <dr::distributed_iterator DI> auto reduce(DI first, DI last) {
  return __detail::reduce(0, false, rng::subrange(first, last), std::plus<>{});
}

} // namespace dr::mhp
