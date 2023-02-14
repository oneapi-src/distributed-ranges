// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/shp/containers/index.hpp>
#include <dr/shp/containers/matrix_entry.hpp>
#include <iterator>

#include "dense_column_view.hpp"
#include "dense_row_view.hpp"

namespace shp {

template <typename T, typename Iter> class dense_matrix_view_accessor {
public:
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  using scalar_type = std::iter_value_t<Iter>;
  using scalar_reference = std::iter_reference_t<Iter>;

  using value_type = shp::matrix_entry<scalar_type, std::size_t>;

  using reference = shp::matrix_ref<T, std::size_t, scalar_reference>;

  using iterator_category = std::random_access_iterator_tag;

  using iterator_accessor = dense_matrix_view_accessor;
  using const_iterator_accessor = iterator_accessor;
  using nonconst_iterator_accessor = iterator_accessor;

  using key_type = shp::index<>;

  constexpr dense_matrix_view_accessor() noexcept = default;
  constexpr ~dense_matrix_view_accessor() noexcept = default;
  constexpr dense_matrix_view_accessor(
      const dense_matrix_view_accessor &) noexcept = default;
  constexpr dense_matrix_view_accessor &
  operator=(const dense_matrix_view_accessor &) noexcept = default;

  constexpr dense_matrix_view_accessor(Iter data, key_type idx,
                                       key_type matrix_shape,
                                       size_type ld) noexcept
      : data_(data), idx_(idx), matrix_shape_(matrix_shape), ld_(ld),
        idx_offset_({0, 0}) {}

  constexpr dense_matrix_view_accessor(Iter data, key_type idx,
                                       key_type idx_offset,
                                       key_type matrix_shape,
                                       size_type ld) noexcept
      : data_(data), idx_(idx), idx_offset_(idx_offset),
        matrix_shape_(matrix_shape), ld_(ld) {}

  constexpr dense_matrix_view_accessor &
  operator+=(difference_type offset) noexcept {
    size_type new_idx = get_global_idx() + offset;
    idx_ = {new_idx / matrix_shape_[1], new_idx % matrix_shape_[1]};

    return *this;
  }

  constexpr bool operator==(const iterator_accessor &other) const noexcept {
    return idx_ == other.idx_;
  }

  constexpr difference_type
  operator-(const iterator_accessor &other) const noexcept {
    return difference_type(get_global_idx()) - other.get_global_idx();
  }

  constexpr bool operator<(const iterator_accessor &other) const noexcept {
    if (idx_[0] < other.idx_[0]) {
      return true;
    } else if (idx_[0] == other.idx_[0]) {
      return idx_[1] < other.idx_[1];
    } else {
      return false;
    }
  }

  constexpr reference operator*() const noexcept {
    return reference(
        key_type(idx_[0] + idx_offset_[0], idx_[1] + idx_offset_[1]),
        data_[idx_[0] * ld_ + idx_[1]]);
  }

private:
  size_type get_global_idx() const noexcept {
    return idx_[0] * matrix_shape_[1] + idx_[1];
  }

private:
  key_type idx_;
  key_type matrix_shape_;
  size_type ld_;

  key_type idx_offset_;

  Iter data_;
};

template <typename T, typename Iter>
using dense_matrix_view_iterator =
    lib::iterator_adaptor<dense_matrix_view_accessor<T, Iter>>;

template <typename T, typename Iter = T *> class dense_matrix_view {
public:
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  using scalar_reference = std::iter_reference_t<Iter>;

  using key_type = shp::index<>;
  using map_type = T;

  using iterator = dense_matrix_view_iterator<T, Iter>;

  dense_matrix_view(Iter data, key_type shape, size_type ld, size_type rank)
      : data_(data), shape_(shape), idx_offset_(key_type{0, 0}), ld_(ld),
        rank_(rank) {}

  dense_matrix_view(Iter data, key_type shape, key_type idx_offset,
                    size_type ld, size_type rank)
      : data_(data), shape_(shape), idx_offset_(idx_offset), ld_(ld),
        rank_(rank) {}

  key_type shape() const noexcept { return shape_; }

  size_type size() const noexcept { return shape()[0] * shape()[1]; }

  scalar_reference operator[](key_type idx) const {
    return data_[idx[0] * ld_ + idx[1]];
  }

  iterator begin() const {
    return iterator(data_, key_type{0, 0}, idx_offset_, shape_, ld_);
  }

  iterator end() const {
    return iterator(data_, key_type{shape_[0], 0}, idx_offset_, shape_, ld_);
  }

  auto row(size_type idx) const {
    return dense_matrix_row_view(data_ + idx * ld_, idx, shape()[1]);
  }

  auto column(size_type idx) const {
    return dense_matrix_column_view(data_ + idx, idx, shape()[0], ld_);
  }

  iterator data() const { return data_; }

  std::size_t rank() const { return rank_; }

private:
  Iter data_;
  key_type shape_;
  key_type idx_offset_;
  size_type ld_;
  size_type rank_;
};

template <std::random_access_iterator Iter>
dense_matrix_view(Iter, shp::index<>, std::size_t)
    -> dense_matrix_view<std::iter_value_t<Iter>, Iter>;

template <std::random_access_iterator Iter>
dense_matrix_view(Iter, shp::index<>)
    -> dense_matrix_view<std::iter_value_t<Iter>, Iter>;

} // namespace shp
