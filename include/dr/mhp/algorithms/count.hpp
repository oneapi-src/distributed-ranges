// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

namespace dr::mhp::__detail {

inline auto add_counts(rng::forward_range auto &&r) {
  rng::range_difference_t<decltype(r)> zero{};
  
  return std::accumulate(rng::begin(r), rng::end(r), zero);
}

inline auto std_count_if(rng::forward_range auto &&r, auto &&pred) {
  using count_type = rng::range_difference_t<decltype(r)>;

  if (rng::empty(r)) {
    return count_type{};
  }

  return std::count_if(std::execution::par_unseq, 
                       dr::__detail::direct_iterator(rng::begin(r)),
                       dr::__detail::direct_iterator(rng::end(r)), 
                       pred);
}

inline auto dpl_count_if(rng::forward_range auto &&r, auto &&pred) {
  using count_type = rng::range_difference_t<decltype(r)>;

#ifdef SYCL_LANGUAGE_VERSION
  if (rng::empty(r)) {
    return count_type{};
  }

  return std::count_if(dpl_policy(),
                       dr::__detail::direct_iterator(rng::begin(r)),
                       dr::__detail::direct_iterator(rng::end(r)),
                       pred);
#else
  assert(false);
  return count_type{};
#endif
}

template <dr::distributed_range DR>
auto count_if(std::size_t root, bool root_provided, DR &&dr, auto &&pred) {
  using count_type = rng::range_difference_t<decltype(dr)>;
  auto comm = default_comm();

  if (rng::empty(dr)) {
    return count_type{};
  }

  if (aligned(dr)) {
    dr::drlog.debug("Parallel count\n");

    // Count within the local segments
    auto count = [=](auto &&r) {
      assert(rng::size(r) > 0);
      if (mhp::use_sycl()) {
        dr::drlog.debug("  with DPL\n");
        return dpl_count_if(r, pred);
      } else {
        dr::drlog.debug("  with CPU\n");
        return std_count_if(r, pred);
      }
    };

    auto locals = rng::views::transform(local_segments(dr), count);
    auto local = add_counts(locals);

    std::vector<count_type> all(comm.size());
    if (root_provided) {
      // Everyone gathers to root, only root adds up the counts
      comm.gather(local, std::span{all}, root);
      if (root == comm.rank()) {
        return add_counts(all);
      } else {
        return count_type{};
      }
    } else {
      // Everyone gathers and everyone adds up the counts
      comm.all_gather(local, all);
      return add_counts(all);
    }
  } else {
    dr::drlog.debug("Serial count\n");
    count_type result{};
    if (!root_provided || root == comm.rank()) {
      result = add_counts(dr);
    }
    barrier();
    return result;
  }
}
    
} // namespace dr::mhp::__detail

namespace dr::mhp {

//
// Ranges
//

// range, elem, w/wo root

template <typename T, dr::distributed_range DR>
auto count(std::size_t root, DR &&dr, const T& value) {
    auto pred = [=](auto &&v) { return v == value; };
    return __detail::count_if(root, true, dr, pred);
}

template <typename T, dr::distributed_range DR>
auto count(DR &&dr, const T& value) {
    auto pred = [=](auto &&v) { return v == value; };
    return __detail::count_if(0, false, dr, pred);
}

// range, predicate, w/wo root

template <dr::distributed_range DR>
auto count_if(std::size_t root, DR &&dr, auto &&pred) {
    return __detail::count_if(root, true, dr, pred);
}

template <dr::distributed_range DR>
auto count_if(DR &&dr, auto &&pred) {
    return __detail::count_if(0, false, dr, pred);
}

//
// Iterators
//

// range, elem, w/wo root

template <typename T, dr::distributed_iterator DI>
auto count(std::size_t root, DI first, DI last, const T& value) {
    auto pred = [=](auto &&v) { return v == value; };
    return __detail::count_if(root, true, rng::subrange(first, last), pred);
}

template <typename T, dr::distributed_iterator DI>
auto count(DI first, DI last, const T& value) {
    auto pred = [=](auto &&v) { return v == value; };
    return __detail::count_if(0, false, rng::subrange(first, last), pred);
}

// range, predicate, w/wo root

template <dr::distributed_iterator DI>
auto count_if(std::size_t root, DI first, DI last, auto &&pred) {
    return __detail::count_if(root, true, rng::subrange(first, last), pred);
}

template <dr::distributed_iterator DI>
auto count_if(DI first, DI last, auto &&pred) {
    return __detail::count_if(0, false, rng::subrange(first, last), pred);
}

}; // namespace dr::mhp
