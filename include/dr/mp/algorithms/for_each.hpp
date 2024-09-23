// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <algorithm>
#include <execution>
#include <type_traits>
#include <utility>
#include <ranges>

#include <dr/concepts/concepts.hpp>
#include <dr/detail/logger.hpp>
#include <dr/detail/onedpl_direct_iterator.hpp>
#include <dr/detail/ranges_shim.hpp>
#include <dr/detail/sycl_utils.hpp>
#include <dr/mp/global.hpp>
#include <dr/detail/tuple_utils.hpp>

namespace dr::mp {

/// Collective for_each on distributed range
void for_each(dr::distributed_range auto &&dr, auto op) {
  dr::drlog.debug(dr::logger::for_each, "for_each: parallel execution\n");
  if (rng::empty(dr)) {
    return;
  }
  assert(aligned(dr));

  for (const auto &s : local_segments(dr)) {
    if (mp::use_sycl()) {
      dr::drlog.debug("  using sycl\n");

      assert(rng::distance(s) > 0);
#ifdef SYCL_LANGUAGE_VERSION
      dr::__detail::parallel_for(
          dr::mp::sycl_queue(), sycl::range<1>(rng::distance(s)),
          [first = rng::begin(s), op](auto idx) { op(first[idx]); })
          .wait();
#else
      assert(false);
#endif
    } else {
      dr::drlog.debug("  using cpu\n");
      rng::for_each(s, op);
    }
  }
  barrier();
}

/// Collective for_each on iterator/sentinel for a distributed range
template <dr::distributed_iterator DI>
void for_each(DI first, DI last, auto op) {
  mp::for_each(rng::subrange(first, last), op);
}

/// Collective for_each on iterator/sentinel for a distributed range
template <dr::distributed_iterator DI, std::integral I>
DI for_each_n(DI first, I n, auto op) {
  auto last = first;
  rng::advance(last, n);
  mp::for_each(first, last, op);
  return last;
}

namespace __detail {
  template <std::size_t Rank>
  using stencil_index_type = dr::__detail::dr_extents<Rank>;

  void stencil_for_each_extended_1(auto op, stencil_index_type<1> begin, stencil_index_type<1> end, const auto& segs) {
    auto [seg0_begin, seg0_end] = std::get<0>(segs).stencil(begin, end);

    auto sub = [](auto a) { return std::get<1>(a) - std::get<0>(a); };
    auto is_zero = [](auto a) { return a != 0; };

    auto zipped = zip_view(seg0_begin, seg0_end);
    auto distance = zipped | std::views::transform(sub);

    if ((distance | std::views::filter(is_zero)).empty())
      return;

    auto seg_infos = dr::__detail::tuple_transform(segs, [begin](auto &&seg) {
      return std::make_pair(seg.begin() + seg.begin_stencil(begin)[0], seg.extents());
    });

    auto do_point = [seg_infos, op](auto index) {
      auto stencils =
          dr::__detail::tuple_transform(seg_infos, [index](auto seg_info) {
            return md::mdspan(
                std::to_address(dr::ranges::local(seg_info.first + index)),
                seg_info.second
            );
          });
      op(stencils, index);
    };
    if (mp::use_sycl()) {
      dr::drlog.debug("  using sycl\n");

#ifdef SYCL_LANGUAGE_VERSION
      dr::__detail::parallel_for(
          dr::mp::sycl_queue(), sycl::range<1>(distance[0]),
          do_point)
          .wait();
#else
      assert(false);
#endif
    } else {
      dr::drlog.debug("  using cpu\n");
      for (std::size_t i = 0; i < distance[0]; i++) {
        do_point(i);
      }
    }
  }

  void stencil_for_each_extended_2(auto op, stencil_index_type<1> begin, stencil_index_type<1> end, const auto& segs) {
    auto [seg0_begin, seg0_end] = std::get<0>(segs).stencil(begin, end);

    auto sub = [](auto a) { return std::get<1>(a) - std::get<0>(a); };
    auto is_zero = [](auto a) { return a != 0; };

    auto zipped = zip_view(seg0_begin, seg0_end);
    auto distance = zipped | std::views::transform(sub);

    if ((distance | std::views::filter(is_zero)).empty())
      return;

    auto seg_infos = dr::__detail::tuple_transform(segs, [begin](auto &&seg) {
      return std::make_pair(seg.begin() + seg.begin_stencil(begin)[0], seg.extents());
    });

    auto do_point = [seg_infos, op](auto index) {
      auto stencils =
          dr::__detail::tuple_transform(seg_infos, [index](auto seg_info) {
            return md::mdspan(
                std::to_address(dr::ranges::local(seg_info.first + index)),
                seg_info.second
            );
          });
      op(stencils, index);
    };
    if (mp::use_sycl()) {
      dr::drlog.debug("  using sycl\n");

#ifdef SYCL_LANGUAGE_VERSION
      dr::__detail::parallel_for(
          dr::mp::sycl_queue(), sycl::range<2>(distance[0], distance[1]),
          do_point)
          .wait();
#else
      assert(false);
#endif
    } else {
      dr::drlog.debug("  using cpu\n");
      for (std::size_t i = 0; i < distance[0]; i++) {
        for (std::size_t i = 0; i < distance[1]; i++) {
          do_point(i);
        }
      }
    }
  }
}

template <std::size_t Rank, typename... Ts>
requires (1 <= Rank && Rank <= 3)
void stencil_for_each_extended(auto op, __detail::stencil_index_type<Rank> begin, __detail::stencil_index_type<Rank> end, dr::distributed_range auto &&...drs) {
  dr::drlog.debug(dr::logger::for_each, "for_each_extended: parallel execution\n");
  auto ranges = std::tie(drs...);
  auto &&dr0 = std::get<0>(ranges);
  if (rng::empty(dr0)) {
    return;
  }

  auto all_segments = rng::views::zip(dr::ranges::segments(drs)...);
  for (const auto &segs : all_segments) {
    if constexpr (Rank == 1) {
      __detail::stencil_for_each_extended_1(op, begin, end, segs);
    }
    else if constexpr (Rank == 2) {
      __detail::stencil_for_each_extended_2(op, begin, end, segs);
    }
    else if constexpr (Rank == 3) {}
  }
  barrier();
}

} // namespace dr::mp
