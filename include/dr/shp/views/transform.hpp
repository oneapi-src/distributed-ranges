// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/concepts/concepts.hpp>
#include <dr/details/ranges_shim.hpp>
#include <dr/shp/normal_distributed_iterator.hpp>

namespace shp {

template <rng::forward_range V, std::copy_constructible F>
class transform_view : public rng::view_interface<transform_view<V, F>> {
public:
  template <rng::viewable_range R>
  transform_view(R &&r, F fn)
      : base_(rng::views::all(std::forward<R>(r))), fn_(fn) {}

  auto begin() const {
    return normal_distributed_iterator<decltype(segments())>(segments(),
                                                             std::size_t(0), 0);
  }

  auto end() const {
    auto segs = segments();
    return normal_distributed_iterator<decltype(segments())>(
        std::move(segs), std::size_t(segs.size()), 0);
  }

  auto segments() const {
    return lib::ranges::segments(base_) |
           rng::views::transform([=](auto &&segment) {
             return segment | rng::views::transform(fn_);
           });
  }

  V base() const { return base_; }

private:
  V base_;
  F fn_;
};

template <rng::viewable_range R, std::copy_constructible F>
transform_view(R &&r, F fn) -> transform_view<rng::views::all_t<R>, F>;

namespace views {

template <std::copy_constructible F> class transform_adapter_closure {
public:
  transform_adapter_closure(F fn) : fn_(fn) {}

  template <rng::viewable_range R> auto operator()(R &&r) const {
    return shp::transform_view(std::forward<R>(r), fn_);
  }

  template <rng::viewable_range R>
  friend auto operator|(R &&r, const transform_adapter_closure &closure) {
    return closure(std::forward<R>(r));
  }

private:
  F fn_;
};

class transform_fn_ {
public:
  template <rng::viewable_range R, std::copy_constructible F>
  auto operator()(R &&r, F &&f) const {
    return transform_adapter_closure(std::forward<F>(f))(std::forward<R>(r));
  }

  template <std::copy_constructible F> auto operator()(F &&fn) const {
    return transform_adapter_closure(std::forward<F>(fn));
  }
};

inline constexpr auto transform = transform_fn_{};
} // namespace views

} // namespace shp
