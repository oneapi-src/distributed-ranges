// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

namespace dr::mp::__detail {

inline auto add_counts(rng::forward_range auto &&r) {
  rng::range_difference_t<decltype(r)> zero{};

  return std::accumulate(rng::begin(r), rng::end(r), zero);
}

inline auto count_if_local(rng::forward_range auto &&r, auto &&pred) {
  if (mp::use_sycl()) {
    dr::drlog.debug("  with DPL\n");
#ifdef SYCL_LANGUAGE_VERSION
    return std::count_if(mp::dpl_policy(),
                         dr::__detail::direct_iterator(rng::begin(r)),
                         dr::__detail::direct_iterator(rng::end(r)), pred);
#else
    assert(false);
#endif
  } else {
    dr::drlog.debug("  with CPU\n");
    return std::count_if(std::execution::par_unseq,
                         dr::__detail::direct_iterator(rng::begin(r)),
                         dr::__detail::direct_iterator(rng::end(r)), pred);
  }
}

template <dr::distributed_range DR>
auto count_if(std::size_t root, bool root_provided, DR &&dr, auto &&pred) {
  using count_type = rng::range_difference_t<decltype(dr)>;
  auto comm = mp::default_comm();

  if (rng::empty(dr)) {
    return count_type{};
  }

  dr::drlog.debug("Parallel count\n");

  // Count within the local segments
  auto count = [=](auto &&r) {
    assert(rng::size(r) > 0);
    return count_if_local(r, pred);
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
}

} // namespace dr::mp::__detail

namespace dr::mp {

class count_fn_ {
public:
  template <typename T, dr::distributed_range DR>
  auto operator()(std::size_t root, DR &&dr, const T &value) const {
    auto pred = [=](auto &&v) { return v == value; };
    return __detail::count_if(root, true, dr, pred);
  }

  template <typename T, dr::distributed_range DR>
  auto operator()(DR &&dr, const T &value) const {
    auto pred = [=](auto &&v) { return v == value; };
    return __detail::count_if(0, false, dr, pred);
  }

  template <typename T, dr::distributed_iterator DI>
  auto operator()(std::size_t root, DI first, DI last, const T &value) const {
    auto pred = [=](auto &&v) { return v == value; };
    return __detail::count_if(root, true, rng::subrange(first, last), pred);
  }

  template <typename T, dr::distributed_iterator DI>
  auto operator()(DI first, DI last, const T &value) const {
    auto pred = [=](auto &&v) { return v == value; };
    return __detail::count_if(0, false, rng::subrange(first, last), pred);
  }
};

inline constexpr count_fn_ count;

class count_if_fn_ {
public:
  template <dr::distributed_range DR>
  auto operator()(std::size_t root, DR &&dr, auto &&pred) const {
    return __detail::count_if(root, true, dr, pred);
  }

  template <dr::distributed_range DR>
  auto operator()(DR &&dr, auto &&pred) const {
    return __detail::count_if(0, false, dr, pred);
  }

  template <dr::distributed_iterator DI>
  auto operator()(std::size_t root, DI first, DI last, auto &&pred) const {
    return __detail::count_if(root, true, rng::subrange(first, last), pred);
  }

  template <dr::distributed_iterator DI>
  auto operator()(DI first, DI last, auto &&pred) const {
    return __detail::count_if(0, false, rng::subrange(first, last), pred);
  }
};

inline constexpr count_if_fn_ count_if;

}; // namespace dr::mp
