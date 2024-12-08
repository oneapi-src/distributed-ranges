// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <algorithm>
#include <execution>
#include <ranges>
#include <type_traits>
#include <utility>

#include <dr/concepts/concepts.hpp>
#include <dr/detail/logger.hpp>
#include <dr/detail/onedpl_direct_iterator.hpp>
#include <dr/detail/ranges_shim.hpp>
#include <dr/detail/sycl_utils.hpp>
#include <dr/detail/tuple_utils.hpp>
#include <dr/mp/global.hpp>

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

void stencil_for_each_extended_1(auto op, stencil_index_type<1> begin,
                                 stencil_index_type<1> end, const auto &segs) {
  auto [seg0_begin, seg0_end] = std::get<0>(segs).stencil(begin, end);

  auto sub = [](auto a) { return std::get<1>(a) - std::get<0>(a); };
  auto is_zero = [](auto a) { return a != 0; };

  auto zipped = zip_view(seg0_begin, seg0_end);
  auto distance = zipped | std::views::transform(sub);

  if (rng::empty(distance | std::views::filter(is_zero)))
    return;

  auto seg_infos = dr::__detail::tuple_transform(segs, [begin](auto &&seg) {
    return std::make_pair(seg.begin() + seg.begin_stencil(begin)[0],
                          seg.extents());
  });

  auto do_point = [seg_infos, op](auto index) {
    auto stencils =
        dr::__detail::tuple_transform(seg_infos, [index](auto seg_info) {
          return md::mdspan(
              std::to_address(dr::ranges::local(seg_info.first + index)),
              seg_info.second);
        });
    op(stencils);
  };
  if (mp::use_sycl()) {
#ifdef SYCL_LANGUAGE_VERSION
    std::cout << "do parallel_for 1d - sycl\n";
    dr::__detail::parallel_for(dr::mp::sycl_queue(),
                               sycl::range<1>(distance[0]), do_point)
        .wait();
#else
    std::cout << "do parallel_for 1d - sycl failed\n";
    assert(false);
#endif
  } else {
    std::cout << "do parallel_for 1d - no sycl\n";
    for (std::size_t i = 0; i < distance[0]; i++) {
      do_point(i);
    }
  }
}

void stencil_for_each_extended_2(auto op, stencil_index_type<2> &begin,
                                 stencil_index_type<2> end, const auto &segs) {
  auto [seg0_begin, seg0_end] = std::get<0>(segs).stencil(begin, end);

  auto sub = [](auto a) {
    auto x = std::get<0>(a);
    auto y = std::get<1>(a);
    return y > x ? y - x : 0;
  };
  auto is_zero = [](auto a) { return a != 0; };

  auto zipped = zip_view(seg0_begin, seg0_end);
  auto distance = zipped | std::views::transform(sub);

  if (rng::empty(distance | std::views::filter(is_zero)))
    return;

  auto seg_infos = dr::__detail::tuple_transform(segs, [&begin](auto &&seg) {
    auto ext = seg.root_mdspan().extents();
    auto begin_stencil = seg.begin_stencil(begin);
    return std::make_pair(md::mdspan(std::to_address(&seg.mdspan_extended()(
                                         begin_stencil[0], begin_stencil[1])),
                                     ext),
                          ext);
  });

  auto do_point = [seg_infos, op](auto index) {
    auto stencils =
        dr::__detail::tuple_transform(seg_infos, [index](auto seg_info) {
          return md::mdspan(
              std::to_address(&seg_info.first(index[0], index[1])),
              seg_info.second);
        });
    op(stencils);
  };
  if (mp::use_sycl()) {
#ifdef SYCL_LANGUAGE_VERSION
    std::cout << "do parallel_for 2d - sycl\n";
    dr::__detail::parallel_for(dr::mp::sycl_queue(),
                               sycl::range<2>(distance[0], distance[1]),
                               do_point)
        .wait();
#else
    std::cout << "do parallel_for 2d - sycl fail\n";
    assert(false);
#endif
  } else {
    std::cout << "do parallel_for 2d - no sycl\n";
    for (std::size_t i = 0; i < distance[0]; i++) {
      for (std::size_t j = 0; j < distance[1]; j++) {
        do_point(stencil_index_type<2>{i, j});
      }
    }
  }
}

void stencil_for_each_extended_3(auto op, stencil_index_type<3> &begin,
                                 stencil_index_type<3> end, const auto &segs) {
  auto [seg0_begin, seg0_end] = std::get<0>(segs).stencil(begin, end);

  auto sub = [](auto a) {
    auto x = std::get<0>(a);
    auto y = std::get<1>(a);
    return y > x ? y - x : 0;
  };
  auto is_zero = [](auto a) { return a != 0; };

  auto zipped = zip_view(seg0_begin, seg0_end);
  auto distance = zipped | std::views::transform(sub);

  if (rng::empty(distance | std::views::filter(is_zero)))
    return;

  auto seg_infos = dr::__detail::tuple_transform(segs, [&begin](auto &&seg) {
    auto ext = seg.root_mdspan().extents();
    auto begin_stencil = seg.begin_stencil(begin);
    return std::make_pair(
        md::mdspan(std::to_address(&seg.mdspan_extended()(
                       begin_stencil[0], begin_stencil[1], begin_stencil[2])),
                   ext),
        ext);
  });

  auto do_point = [seg_infos, op](auto index) {
    auto stencils =
        dr::__detail::tuple_transform(seg_infos, [index](auto seg_info) {
          return md::mdspan(
              std::to_address(&seg_info.first(index[0], index[1], index[2])),
              seg_info.second);
        });
    op(stencils);
  };
  if (mp::use_sycl()) {
#ifdef SYCL_LANGUAGE_VERSION
    dr::__detail::parallel_for(
        dr::mp::sycl_queue(),
        sycl::range<3>(distance[0], distance[1], distance[2]), do_point)
        .wait();
#else
    assert(false);
#endif
  } else {
    for (std::size_t i = 0; i < distance[0]; i++) {
      for (std::size_t j = 0; j < distance[1]; j++) {
        for (std::size_t k = 0; k < distance[3]; k++) {
          do_point(stencil_index_type<3>{i, j, k});
        }
      }
    }
  }
}
} // namespace __detail

template <std::size_t Rank, typename... Ts>
requires(1 <= Rank && Rank <= 3) void stencil_for_each_extended(
    auto op, __detail::stencil_index_type<Rank> begin,
    __detail::stencil_index_type<Rank> end,
    dr::distributed_range auto &&...drs) {
  auto ranges = std::tie(drs...);
  auto &&dr0 = std::get<0>(ranges);
  if (rng::empty(dr0)) {
    return;
  }

  auto all_segments = rng::views::zip(dr::ranges::segments(drs)...);
  for (const auto &segs : all_segments) {
    if constexpr (Rank == 1) {
      __detail::stencil_for_each_extended_1(op, begin, end, segs);
    } else if constexpr (Rank == 2) {
      __detail::stencil_for_each_extended_2(op, begin, end, segs);
    } else if constexpr (Rank == 3) {
      __detail::stencil_for_each_extended_3(op, begin, end, segs);
    } else {
      static_assert(false, "Not supported"); // sycl for_each does not support
                                             // more than 3 dimensions
    }
  }
  barrier();
}

} // namespace dr::mp
