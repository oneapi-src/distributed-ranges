// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/detail/mdspan_shim.hpp>
#include <dr/detail/ranges_shim.hpp>

namespace dr::mhp {

//
// Add a local mdspan to the underlying segment
//
template <typename BaseSegment, typename Extents,
          typename Layout = md::layout_right>
class mdsegment : public BaseSegment {
public:
  mdsegment(BaseSegment segment, Extents local_extents)
      : BaseSegment(segment), mdspan_(local_mdspan(segment, local_extents)) {}

  auto mdspan() const { return mdspan_; }

private:
  using T = rng::range_value_t<BaseSegment>;

  static auto local_mdspan(BaseSegment segment, Extents local_extents) {
    // Undefined behavior if the segments is not local
    T *ptr = dr::ranges::rank(segment) == default_comm().rank()
                 ? std::to_address(dr::ranges::local(rng::begin(segment)))
                 : nullptr;
    return md::mdspan(ptr, local_extents);
  }

  md::mdspan<T, Extents, Layout> mdspan_;
};

//
// Mdspan maps a multi-dimensional index into a linear offset, and
// then uses this to access the underlying distributed range
//
template <typename Iter> class distributed_accessor {
public:
  using data_handle_type = Iter;
  using reference = std::iter_reference_t<Iter>;
  using offset_policy = distributed_accessor;

  constexpr distributed_accessor() noexcept = default;
  constexpr auto access(Iter iter, std::size_t index) const {
    return iter[index];
  }

  constexpr auto offset(Iter iter, std::size_t index) const noexcept {
    return iter + index;
  }
};

//
// Wrap a distributed range, adding an mdspan and adapting the
// segments to also be mdspans for local access
//
template <distributed_contiguous_range R, typename Extents,
          typename Layout = md::layout_right>
class mdspan_view : public rng::view_interface<mdspan_view<R, Extents>> {
private:
  using base_type = rng::views::all_t<R>;
  using iterator_type = rng::iterator_t<base_type>;
  using mdspan_type = md::mdspan<iterator_type, Extents, Layout,
                                 distributed_accessor<iterator_type>>;
  using difference_type = rng::iter_difference_t<iterator_type>;

public:
  mdspan_view(R r, Extents extents)
      : base_(rng::views::all(r)), mdspan_(rng::begin(base_), extents) {
    // Should be error, not assert
    assert(rng::size(r) == rng::size(mdspan()));
  }

  // Base implements random access range
  auto begin() const { return base_.begin(); }
  auto end() const { return base_.end(); }
  auto operator[](difference_type n) { return base_[n]; }

  // Add a local mdspan to the base segment
  auto segments() const {
    auto make_md = [extents = local_extents()](auto segment) {
      return mdsegment(segment, extents);
    };
    return dr::ranges::segments(base_) | rng::views::transform(make_md);
  }

  // Mdspan access to base
  auto mdspan() const { return mdspan_; }

private:
  auto local_extents() const {
    // Copy extents to array so we can modify it
    std::array<typename Extents::index_type, Extents::rank()> local_extents;
    std::size_t i = 0;
    for (auto &e : local_extents) {
      e = mdspan_.extents().extent(i++);
    }
    // Assume decomposition along leading dimension, and divide the first
    // dimension by number of segments
    local_extents[0] = mdspan_.extents().extent(0) /
                       std::size_t(rng::size(dr::ranges::segments(base_)));
    ;

    return Extents(local_extents);
  }

  base_type base_;
  mdspan_type mdspan_;
};

template <typename R, typename Extents>
mdspan_view(R &&r, Extents extents)
    -> mdspan_view<rng::views::all_t<R>, Extents>;

} // namespace dr::mhp

namespace dr::mhp::views {

template <typename Extents> class mdspan_adapter_closure {
public:
  mdspan_adapter_closure(Extents extents) : extents_(extents) {}

  template <rng::viewable_range R> auto operator()(R &&r) const {
    return mdspan_view(std::forward<R>(r), extents_);
  }

  template <rng::viewable_range R>
  friend auto operator|(R &&r, const mdspan_adapter_closure &closure) {
    return closure(std::forward<R>(r));
  }

private:
  Extents extents_;
};

class mdspan_fn_ {
public:
  template <rng::viewable_range R, typename Extents>
  auto operator()(R &&r, Extents &&extents) const {
    return mdspan_adapter_closure(std::forward<Extents>(extents))(
        std::forward<R>(r));
  }

  template <typename Extents> auto operator()(Extents &&extents) const {
    return mdspan_adapter_closure(std::forward<Extents>(extents));
  }
};

inline constexpr auto mdspan = mdspan_fn_{};

} // namespace dr::mhp::views
