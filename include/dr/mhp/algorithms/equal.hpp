// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once
#include <concepts>
#include <dr/concepts/concepts.hpp>

namespace dr::mhp {
namespace __detail {

inline int std_all(rng::forward_range auto &&r) {
  auto min = [](double x, double y) { return std::min(x, y); };
  return std::reduce(std::execution::par_unseq, rng::begin(r), rng::end(r), 1,
                     min);
}

inline bool check_all(int value) { return value == 1; }
inline bool std_equal(rng::forward_range auto &&r1,
                      rng::forward_range auto &&r2) {
  return std::equal(std::execution::par_unseq, rng::begin(r1), rng::end(r1),
                    rng::begin(r2), rng::end(r2));
}

inline auto dpl_equal(rng::forward_range auto &&r1,
                      rng::forward_range auto &&r2) {
#ifdef SYCL_LANGUAGE_VERSION
  return std::equal(dr::mhp::dpl_policy(),
                    dr::__detail::direct_iterator(rng::begin(r1)),
                    dr::__detail::direct_iterator(rng::end(r1)),
                    dr::__detail::direct_iterator(rng::begin(r2)),
                    dr::__detail::direct_iterator(rng::end(r2)));
#else
  assert(false);
  return false;
#endif
}

} // namespace __detail
template <dr::distributed_range R1, dr::distributed_range R2>
  requires std::equality_comparable_with<rng::range_value_t<R1>,
                                         rng::range_value_t<R2>>
bool equal(std::size_t root, bool root_provided, R1 &&r1, R2 &&r2) {
  // using value_type = rng::range_value_t<DR>;
  auto comm = default_comm();

  if (rng::size(r1) != rng::size(r2)) {
    return false;
  }
  if (rng::empty(r1)) {
    return true;
  }

  if (aligned(r1, r2)) {
    dr::drlog.debug("Parallel equal\n");

    auto compare = [=](auto &&r) {
      assert(rng::size(r.first) > 0);
      assert(rng::size(r.second) > 0);
      bool res = false;
      if (mhp::use_sycl()) {
        dr::drlog.debug("  with DPL\n");
        res = dpl_equal(r.first, r.second);
      } else {
        dr::drlog.debug("  with CPU\n");
        res = std_equal(r.first, r.second);
      }
      if (res) {
        return 1;
      }
      return 0;
    };
    auto locals = rng::views::transform(
        rng::views::zip(local_segments(r1), local_segments(r2)), compare);
    auto local = std_all(locals);

    std::vector<int> all(comm.size());
    // we have to use int here,
    // because std::vector<bool> has custom implementation that
    // does not work with rng::data and cannot be converted to std::span
    if (root_provided) {
      // Everyone gathers to root, only root reduces
      comm.gather(local, std::span{all}, root);
      if (root == comm.rank()) {
        return check_all(std_all(all));
      } else {
        return true;
      }
    } else {
      // Everyone gathers and everyone reduces
      comm.all_gather(local, all);
      return check_all(std_all(all));
    }
  } else {
    dr::drlog.debug("Serial equal\n");
    bool result = true;
    if (!root_provided || root == comm.rank()) {
      result = std_equal(r1, r2);
    }
    barrier();
    return result;
  }
  return true;
}

template <dr::distributed_range R1, dr::distributed_range R2>
  requires std::equality_comparable_with<rng::range_value_t<R1>,
                                         rng::range_value_t<R2>>
bool equal(R1 &&r1, R2 &&r2) {
  return __detail::equal(0, false, r1, r2);
}

} // namespace dr::mhp
