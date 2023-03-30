// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstddef>
#include <iterator>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <dr/details/ranges_shim.hpp>
#include <dr/mhp/alignment.hpp>

namespace mhp {

inline auto select_dist_range(auto &&v, auto &&...rest) {
  if constexpr (lib::distributed_range<decltype(v)>) {
    return rng::views::all(v);
  } else {
    return select_dist_range(rest...);
  }
}

template <typename I>
concept is_zip_iterator =
    std::forward_iterator<I> && requires(I &iter) { std::get<0>(*iter); };

template <rng::viewable_range... R>
class zip_view : public rng::view_interface<zip_view<R...>> {
public:
  zip_view(const zip_view &z) : base_(z.base_) {}
  zip_view(zip_view &&z) : base_(std::move(z.base_)) {}

  template <typename... V> zip_view(V &&...v) : base_(rng::views::all(v)...) {
    auto segments = lib::ranges::segments(select_dist_range(v...));
    segment_descriptor descriptor{0, 0};
    for (auto segment : segments) {
      descriptor.size = rng::distance(segment);
      segment_descriptors_.emplace_back(descriptor);
      descriptor.offset += descriptor.size;
    }
  }

  auto begin() const {
    return lib::normal_distributed_iterator<decltype(segments())>(
        segments(), std::size_t(0), 0);
  }

  auto end() const {
    auto segs = segments();
    return lib::normal_distributed_iterator<decltype(segments())>(
        std::move(segs), std::size_t(rng::distance(segs)), 0);
  }

  auto segments() const {
    auto zip_segments = [this](auto &&...base) {
      auto zip_segment = [](auto &&v) {
        auto zip = [](auto &&...refs) { return rng::views::zip(refs...); };
        return std::apply(zip, v);
      };

      auto z = rng::views::zip(this->base_segments(base)...) |
               rng::views::transform(zip_segment);
      if (aligned(rng::begin(base)...)) {
        return z;
      } else {
        return decltype(z){};
      }
    };

    return std::apply(zip_segments, base_);
  }

  auto base() const { return base_; }

private:
  struct segment_descriptor {
    std::size_t offset, size;
  };

  template <rng::range V>
    requires(lib::is_iota_view_v<std::remove_cvref_t<V>>)
  auto base_segments(V &&base) const {
    auto make_iota_segment = [base](const segment_descriptor &d) {
      return rng::subrange(rng::begin(base) + d.offset,
                           rng::begin(base) + d.offset + d.size);
    };

    return segment_descriptors_ | rng::views::transform(make_iota_segment);
  }

  template <rng::range V> auto base_segments(V &&base) const {
    return lib::ranges::segments(base);
    ;
  }

  std::tuple<R...> base_;
  // For large number of segments, this will be expensive to copy
  std::vector<segment_descriptor> segment_descriptors_;
};

template <rng::viewable_range... R>
zip_view(R &&...r) -> zip_view<rng::views::all_t<R>...>;

namespace views {

template <rng::viewable_range... R> auto zip(R &&...r) {
  return zip_view(std::forward<R>(r)...);
}

} // namespace views

} // namespace mhp

namespace DR_RANGES_NAMESPACE {

template <mhp::is_zip_iterator ZI> auto local_(ZI zi) {
  auto refs_to_local_zip_iterator = [](auto &&...refs) {
    // Convert the first segment of each component to local and then
    // zip them together, returning the begin() of the zip view
    return rng::begin(rng::zip_view(
        (lib::ranges::local(lib::ranges::segments(&refs)[0]))...));
  };
  return std::apply(refs_to_local_zip_iterator, *zi);
}

} // namespace DR_RANGES_NAMESPACE
