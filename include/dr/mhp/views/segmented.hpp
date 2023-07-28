// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/detail/remote_subrange.hpp>

namespace dr::mhp {

template <typename BaseIter, typename SegTplIter, typename SegTplSentinel>
class segmented_view_iterator {
public:
  using iterator_category = std::forward_iterator_tag;
  using difference_type = rng::iter_difference_t<SegTplIter>;
  using value_type = dr::remote_subrange<BaseIter>;

  segmented_view_iterator() {}
  segmented_view_iterator(BaseIter base_begin, SegTplIter tpl_begin,
                          SegTplSentinel tpl_end)
      : base_cur_(base_begin), tpl_cur_(tpl_begin), tpl_end_(tpl_end) {}

  auto operator==(segmented_view_iterator other) const {
    return tpl_cur_ == other.tpl_cur_;
  }
  auto &operator++() {
    base_cur_ += rng::size(*tpl_cur_);
    tpl_cur_++;
    return *this;
  }
  auto operator++(int) {
    auto iter(*this);
    base_cur_ += rng::size(*tpl_cur_);
    tpl_cur_++;
    return iter;
  }
  auto operator*() const {
    return dr::remote_subrange(base_cur_, base_cur_ + rng::size(*tpl_cur_),
                               dr::ranges::rank(*tpl_cur_));
  }

private:
  BaseIter base_cur_;
  SegTplIter tpl_cur_;
  SegTplSentinel tpl_end_;
};

//
// Some distributed algorithms need an iota_view as an operand. An
// iota_view does not depend on external data and can be segmented as
// needed. The segmented_view creates segments for a range using the
// segments of another range. It can be used to create segments for an
// iota_view, using the segments of a distributed_range.
//
// It should be usable if you have a range that is local and
// replicated across all processes, but that is not tested.
//
template <rng::random_access_range R, rng::forward_range SegTpl>
class segmented_view : public rng::view_interface<segmented_view<R, SegTpl>> {
public:
  template <typename V1, typename V2>
  segmented_view(V1 &&r, V2 &&tpl)
      : base_(rng::views::all(std::forward<V1>(r))),
        segments_tpl_(rng::views::all(std::forward<V2>(tpl))) {}

  auto begin() const {
    return segmented_view_iterator(rng::begin(base_), rng::begin(segments_tpl_),
                                   rng::end(segments_tpl_));
  }
  auto end() const {
    return segmented_view_iterator(rng::begin(base_), rng::end(segments_tpl_),
                                   rng::end(segments_tpl_));
  }

  auto size() const { return rng::size(segments_tpl_); }

private:
  rng::views::all_t<R> base_;
  rng::views::all_t<SegTpl> segments_tpl_;
};

template <typename R, typename Seg>
segmented_view(R &&r, Seg &&seg)
    -> segmented_view<rng::views::all_t<R>, rng::views::all_t<Seg>>;

namespace views {

/// Zip
template <typename R, typename Seg> auto segmented(R &&r, Seg &&seg) {
  return segmented_view(std::forward<R>(r), std::forward<Seg>(seg));
}

} // namespace views

} // namespace dr::mhp
