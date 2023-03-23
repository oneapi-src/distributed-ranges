// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "distributed_iterators.hpp"
#include "index.hpp"

namespace mhp {

using key_type = mhp::index<>;

template <typename DM> class dm_segment {
private:
  using iterator = dm_segment_iterator<DM>;
  using value_type = typename DM::value_type;

public:
  using difference_type = std::ptrdiff_t;
  dm_segment() = default;
  dm_segment(DM *dm, value_type *ptr, key_type shape,
             std::size_t segment_index) {
    dm_ = dm;
    ptr_ = ptr;
    shape_ = shape;
    rank_ = segment_index;
    size_ = shape[0] * shape[1];
  }

  auto size() const { return size_; }

  auto begin() const { return iterator(dm_, rank_, 0); }
  auto end() const { return begin() + size(); }

  auto operator[](difference_type n) const { return *(begin() + n); }

  bool is_local() { return rank_ == default_comm().rank(); }
  key_type shape() { return shape_; }

  DM *dm() { return dm_; }

private:
  DM *dm_;
  value_type *ptr_;
  key_type shape_;
  std::size_t rank_;
  std::size_t size_;
};
template <typename DM> class dm_segments : public std::span<dm_segment<DM>> {
public:
  dm_segments() {}
  dm_segments(DM *dm) : std::span<dm_segment<DM>>(dm->segments_) { dm_ = dm; }

private:
  const DM *dm_;
}; // dm_segments

template <typename T> class dm_row_view {
  using dmatrix = distributed_dense_matrix<T>;
  using dmsegment = dm_segment<dmatrix>;

public:
  using iterator = dm_row_iterator<dmatrix>;

  dm_row_view(size_t idx, T *ptr, dmsegment &segment, size_t size)
      : index_(idx), data_(ptr), segmentref_(segment), size_(size){};

  iterator begin() { return iterator(data_, segmentref_.dm()); }
  iterator end() { return begin() + size_; }

  T &operator[](size_t idx) {
    assert(data_ != nullptr);
    assert(idx < size_);
    return data_[idx];
  }

  dmsegment &segment() { return segmentref_; }
  size_t idx() { return index_; }

private:
  size_t index_;
  T *data_ = nullptr;
  dmsegment &segmentref_;
  size_t size_ = 0;
};
template <typename DM>
class dm_rows : public std::vector<dm_row_view<typename DM::value_type>> {
public:
  using iterator = dm_rows_iterator<DM>;

  dm_rows(DM *dm) : dm_(dm){};

  auto &halo() const { return dm_->halo(); }
  auto segments() { return dm_->segments(); }

private:
  DM *dm_ = nullptr;
};

template <typename T, typename Allocator> class distributed_dense_matrix {
public:
  using value_type = T;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using key_type = mhp::index<>;

  using iterator =
      lib::normal_distributed_iterator<dm_segments<distributed_dense_matrix>>;

  distributed_dense_matrix(key_type shape,
                           lib::halo_bounds hb = lib::halo_bounds(),
                           Allocator allocator = Allocator())
      : shape_(shape), grid_shape_(default_comm().size(), 1),
        segment_shape_((shape_[0] + default_comm().size() - 1) /
                           default_comm().size(),
                       shape_[1]),
        segment_size_(std::max({segment_shape_[0] * segment_shape_[1],
                                hb.prev * segment_shape_[1],
                                hb.next * segment_shape_[1]})),
        data_size_(segment_size_ + hb.prev * shape_[1] + hb.next * shape_[1]),
        rows_(this) {
    init_(hb, allocator);
  }

  distributed_dense_matrix(size_t rows, size_t cols,
                           lib::halo_bounds hb = lib::halo_bounds(),
                           Allocator allocator = Allocator())
      : shape_(key_type(rows, cols)), // partition_(new mhp::block_cyclic()),
        grid_shape_(default_comm().size(), 1),
        segment_shape_((shape_[0] + default_comm().size() - 1) /
                           default_comm().size(),
                       shape_[1]),
        segment_size_(std::max({segment_shape_[0] * segment_shape_[1],
                                hb.prev * segment_shape_[1],
                                hb.next * segment_shape_[1]})),
        data_size_(segment_size_ + hb.prev * shape_[1] + hb.next * shape_[1]),
        rows_(this) {

    init_(hb, allocator);
  }
  ~distributed_dense_matrix() {
    fence();
    active_wins().erase(win_.mpi_win());
    win_.free();
    allocator_.deallocate(data_, data_size_);
    data_ = nullptr;
    delete halo_;
  }

  dm_rows<distributed_dense_matrix<T>> local_rows() {
    dm_rows row_view_(this);
    size_t row_index_ = 0;
    for (auto _titr = segments_.begin(); _titr != segments_.end(); ++_titr) {
      if ((*_titr).is_local()) {
        for (size_t s = 0; s < (*_titr).shape()[0]; s++) {
          row_view_.emplace_back(row_index_ + s,
                                 data_ + s * (*_titr).shape()[1], *_titr,
                                 (*_titr).shape()[1]);
        }
      }
      row_index_ += (*_titr).shape()[0];
    };
    return row_view_;
  }

  auto &rows() { return rows_; }

  auto begin() const { return iterator(segments(), 0, 0); }
  auto end() const {
    return iterator(segments(), rng::distance(segments()), 0);
  }

  T *data() { return data(); }
  key_type shape() const noexcept { return shape_; }
  size_type size() const noexcept { return shape()[0] * shape()[1]; }
  dm_segments<distributed_dense_matrix> segments() const {
    return dm_segments_;
  }
  auto &halo() { return *halo_; }
  lib::halo_bounds &halo_bounds() { return halo_bounds_; }

private:
  void init_(lib::halo_bounds hb, auto allocator) {

    data_ = allocator.allocate(data_size_);
    halo_ = new lib::span_halo<T>(default_comm(), data_, data_size_, hb);

    // one segment per node, 1-d arrangement of segments

    segments_.reserve(grid_shape_[0] * grid_shape_[1]);

    size_t idx = 0;
    for (std::size_t i = 0; i < grid_shape_[0]; i++) {
      for (std::size_t j = 0; j < grid_shape_[1]; j++) {
        T *_ptr = (idx == default_comm().rank()) ? data_ : nullptr;

        key_type _ts(
            segment_shape_[0] -
                ((idx + 1) / default_comm().size()) *
                    (default_comm().size() * segment_shape_[0] - shape_[0]),
            segment_shape_[1]);
        segments_.emplace_back(this, _ptr, _ts, idx);
        idx++;
      }
    }

    size_t row_index_ = 0;
    for (auto _titr = segments_.begin(); _titr != segments_.end(); ++_titr) {
      for (size_t s = 0; s < (*_titr).shape()[0]; s++) {
        T *_dataptr =
            (*_titr).is_local() ? (data_ + s * (*_titr).shape()[1]) : nullptr;
        rows_.emplace_back(row_index_++, _dataptr, *_titr, (*_titr).shape()[1]);
      }
    };

    win_.create(default_comm(), data_, data_size_ * sizeof(T));
    active_wins().insert(win_.mpi_win());
    dm_segments_ = dm_segments<distributed_dense_matrix>(this);
    fence();
  }

private:
  friend dm_segment_iterator<distributed_dense_matrix>;
  friend dm_segments<distributed_dense_matrix>;

  key_type shape_;
  key_type grid_shape_; // currently (N, 1)
  key_type segment_shape_;

  std::size_t segment_size_ = 0;
  std::size_t data_size_ = 0;
  T *data_ = nullptr;

  dm_rows<distributed_dense_matrix<T>> rows_;

  lib::span_halo<T> *halo_;
  lib::halo_bounds halo_bounds_;
  std::size_t size_;

  // std::vector<dm_segment<distributed_dense_matrix>> segments_;
  // dm_segments<distributed_dense_matrix> dm_segments_;

  dm_segments<distributed_dense_matrix> dm_segments_;
  std::vector<dm_segment<distributed_dense_matrix>> segments_;
  lib::rma_window win_;
  Allocator allocator_;
};

template <typename T>
void for_each(dm_rows<distributed_dense_matrix<T>> view, auto op) {
  for (auto itr = view.begin(); itr != view.end(); itr++) {
    if ((*itr).segment().is_local()) {
      op(*itr);
    }
  }
};

} // namespace mhp

// Needed to satisfy rng::viewable_range
template <typename T>
inline constexpr bool rng::enable_borrowed_range<
    mhp::dm_segments<mhp::distributed_dense_matrix<T>>> = true;

template <typename T>
inline constexpr bool
    rng::enable_borrowed_range<mhp::dm_rows<mhp::distributed_dense_matrix<T>>> =
        true;
