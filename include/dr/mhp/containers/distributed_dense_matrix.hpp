// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "distributed_iterators.hpp"
#include "index.hpp"
#include "subrange.hpp"

namespace mhp {

using key_type = mhp::index<>;

class distributed_matrix_partition {};
class by_row final : public distributed_matrix_partition {};
class by_column final : public distributed_matrix_partition {};

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
  DM *dm_;
}; // dm_segments

template <typename T> class dm_row : public std::span<T> {
  using dmatrix = mhp::distributed_dense_matrix<T>;
  using dmsegment = mhp::dm_segment<dmatrix>;
  using iterator = typename std::span<T>::iterator;

public:
  dm_row(){};
  dm_row(signed long idx, T *ptr, dmsegment *segment, std::size_t size)
      : std::span<T>({ptr, size}), index_(idx), data_(ptr), segment_(segment),
        size_(size){};

  dmsegment *segment() { return segment_; }
  signed long idx() { return index_; }

  T &operator[](int index) { return *(std::span<T>::begin() + index); }

  dm_row<T> operator=(dm_row<T> other) {
    std::cout << "operator=" << std::endl;
    iterator i = this->begin(), oi = other.begin();
    while (i != this->end()) {
      std::cout << "*(oi++) = *(i++);" << std::endl;
      *(oi++) = *(i++);
    }
    return *this;
  }

private:
  signed long index_ = 0;
  T *data_ = nullptr;
  dmsegment *segment_ = nullptr;
  std::size_t size_ = 0;
};

template <typename DM>
class dm_rows : public std::vector<dm_row<typename DM::value_type>> {
public:
  using iterator = dm_rows_iterator<DM>;
  using value_type = dm_row<typename DM::value_type>;

  dm_rows(DM *dm) { dm_ = dm; }

  auto segments() { return dm_->segments(); }
  auto &halo() { return dm_->halo(); }

  iterator begin() const {
    assert(dm_ != nullptr);
    return dm_rows_iterator(dm_, 0, 0);
  }
  iterator end() const {
    assert(dm_ != nullptr);
    return dm_rows_iterator(dm_, 0, dm_->shape()[1]);
  }

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

  distributed_dense_matrix(
      std::size_t rows, std::size_t cols,
      lib::halo_bounds hb = lib::halo_bounds(),
      distributed_matrix_partition *partition = new by_row(),
      Allocator allocator = Allocator())
      : distributed_dense_matrix(key_type(rows, cols), hb, partition,
                                 allocator){};

  distributed_dense_matrix(
      key_type shape, lib::halo_bounds hb = lib::halo_bounds(),
      distributed_matrix_partition *partition = new by_row(),
      Allocator allocator = Allocator())
      : shape_(shape), // partition_(new mhp::block_cyclic()),
        segment_shape_((shape_[0] + default_comm().size() - 1) /
                           default_comm().size(),
                       shape_[1]),
        segment_size_(std::max({segment_shape_[0] * segment_shape_[1],
                                hb.prev * segment_shape_[1],
                                hb.next * segment_shape_[1]})),
        data_size_(segment_size_ + hb.prev * segment_shape_[1] +
                   hb.next * segment_shape_[1]),
        dm_rows_(this) {

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

  dm_rows<distributed_dense_matrix> &rows() { return dm_rows_; }

  iterator begin() const { return iterator(segments(), 0, 0); }
  iterator end() const {
    return iterator(segments(), rng::distance(segments()), 0);
  }

  T *data() { return data_; }
  key_type shape() noexcept { return shape_; }
  size_type size() noexcept { return shape()[0] * shape()[1]; }
  auto segments() { return dm_segments_; }
  size_type segsize() { return segment_size_; }

  auto &halo() { return *halo_; }
  lib::halo_bounds &halo_bounds() { return halo_bounds_; }

private:
  void init_(lib::halo_bounds hb, auto allocator) {

    halo_bounds_ = hb;
    data_ = allocator.allocate(data_size_);

    grid_size_ = default_comm().size();

    hb.prev *= shape_[0];
    hb.next *= shape_[0];
    halo_ = new lib::span_halo<T>(default_comm(), data_, data_size_, hb);

    // prepare segments
    // one segment per node, 1-d arrangement of segments

    segments_.reserve(grid_size_);

    std::size_t idx = 0;
    for (std::size_t i = 0; i < grid_size_; i++) {
      T *_ptr = (idx == default_comm().rank()) ? data_ : nullptr;
      key_type _ts(
          segment_shape_[0] -
              ((idx + 1) / default_comm().size()) *
                  (default_comm().size() * segment_shape_[0] - shape_[0]),
          segment_shape_[1]);
      segments_.emplace_back(this, _ptr, _ts, idx);
      idx++;
    }

    // prepare all rows
    dm_rows_.reserve(segment_shape_[0]);

    std::size_t row_index_ = 0;
    for (auto _titr = segments_.begin(); _titr != segments_.end(); ++_titr) {
      for (std::size_t s = 0; s < (*_titr).shape()[0]; s++) {
        T *_dataptr =
            (*_titr).is_local() ? (data_ + s * (*_titr).shape()[1]) : nullptr;
        dm_rows_.emplace_back(row_index_++, _dataptr, &(*_titr),
                              (*_titr).shape()[1]);
      }
    };

    win_.create(default_comm(), data_, data_size_ * sizeof(T));
    active_wins().insert(win_.mpi_win());
    dm_segments_ = dm_segments<distributed_dense_matrix>(this);
    // dm_rows_ = dm_rows<distributed_dense_matrix>(this);
    fence();
  }

private:
  friend dm_segment_iterator<distributed_dense_matrix>;
  friend dm_segments<distributed_dense_matrix>;
  friend dm_rows<distributed_dense_matrix>;
  friend dm_rows_iterator<distributed_dense_matrix>;

  key_type shape_;
  std::size_t grid_size_; // currently (N, 1)
  key_type segment_shape_;

  std::size_t segment_size_ = 0; // size of local data
  std::size_t data_size_ = 0;    // all data with halo buffers

  T *data_ = nullptr; // local data ptr

  lib::span_halo<T> *halo_;
  lib::halo_bounds halo_bounds_;
  std::size_t size_;

  dm_segments<distributed_dense_matrix> dm_segments_;
  dm_rows<distributed_dense_matrix<T>> dm_rows_;

  std::vector<dm_segment<distributed_dense_matrix>> segments_;

  lib::rma_window win_;
  Allocator allocator_;
}; // class distributed_dense_matrix

// template <typename T>
// void for_each(dm_rows<distributed_dense_matrix<T>>::iterator First,
//               dm_rows<distributed_dense_matrix<T>>::iterator Last, auto op) {
//   for (auto itr = First; itr != Last; itr++) {
//     if ((*itr).segment().is_local()) {
//       op(*itr);
//     }
//   }
// };

template <typename T>
void for_each(dm_rows<distributed_dense_matrix<T>> Rng, auto op) {
  for (auto itr = Rng.begin(); itr != Rng.end(); itr++) {
    dm_row<T> &r = *itr;
    if (r.segment()->is_local()) {
      op(*itr);
    }
  }
};

template <typename DM> auto &halo(dm_rows<DM> &&dr) { return dr.halo(); }

} // namespace mhp

// Needed to satisfy rng::viewable_range
template <typename T>
inline constexpr bool rng::enable_borrowed_range<
    mhp::dm_segments<mhp::distributed_dense_matrix<T>>> = true;

template <typename T>
inline constexpr bool
    rng::enable_borrowed_range<mhp::dm_rows<mhp::distributed_dense_matrix<T>>> =
        true;
