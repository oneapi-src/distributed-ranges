// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "index.hpp"
#include "segments.hpp"
#include "rows.hpp"
#include "subrange.hpp"

namespace dr::mhp {

using key_type = index<>;

class distributed_matrix_partition {};
class by_row final : public distributed_matrix_partition {};
class by_column final : public distributed_matrix_partition {};

template <typename DM> class dm_segments : public std::span<d_segment<DM>> {
public:
  dm_segments() {}
  dm_segments(DM *dm) : std::span<d_segment<DM>>(dm->segments_) { dm_ = dm; }

private:
  DM *dm_;
}; // dm_segments

template <typename T, typename Allocator> class distributed_dense_matrix {
public:
  using value_type = T;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using key_type = index<>;

  using iterator =
      dr::normal_distributed_iterator<dm_segments<distributed_dense_matrix>>;

  distributed_dense_matrix(
      std::size_t rows, std::size_t cols,
      dr::halo_bounds hb = dr::halo_bounds(),
      distributed_matrix_partition *partition = new by_row(),
      Allocator allocator = Allocator())
      : distributed_dense_matrix(key_type(rows, cols), hb, partition,
                                 allocator){};

  distributed_dense_matrix(
      std::size_t rows, std::size_t cols, T fillval,
      dr::halo_bounds hb = dr::halo_bounds(),
      distributed_matrix_partition *partition = new by_row(),
      Allocator allocator = Allocator())
      : distributed_dense_matrix(key_type(rows, cols), hb, partition,
                                 allocator) {

    for (std::size_t _i = 0; _i < data_size_; _i++)
      data_[_i] = fillval;
  };

  distributed_dense_matrix(
      key_type shape, dr::halo_bounds hb = dr::halo_bounds(),
      distributed_matrix_partition *partition = new by_row(),
      Allocator allocator = Allocator())
      : shape_(shape), segment_shape_((shape_[0] + default_comm().size() - 1) /
                                          default_comm().size(),
                                      shape_[1]),
        segment_size_(std::max({segment_shape_[0] * segment_shape_[1],
                                hb.prev * segment_shape_[1],
                                hb.next * segment_shape_[1]})),
        data_size_(segment_size_ + hb.prev * segment_shape_[1] +
                   hb.next * segment_shape_[1]),
        dm_rows_(this), dm_halop_rows_(this), dm_halon_rows_(this) {
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
  size_type segment_size() { return segment_size_; }
  key_type segment_shape() { return segment_shape_; }

  auto &halo() { return *halo_; }
  dr::halo_bounds &halo_bounds() { return halo_bounds_; }

  bool is_local_row(int index) {
    if (index >= local_rows_indices_.first &&
        index <= local_rows_indices_.second)
      return true;
    return false;
  }
  // index of cell on linear view
  bool is_local_cell(int index) {
    if (index >= local_rows_indices_.first * (int)shape_[1] &&
        index < (local_rows_indices_.second + 1) * (int)shape_[1]) {
      return true;
    }
    return false;
  }

  std::pair<int, int> local_rows_indices() { return local_rows_indices_; }

  // for debug only
  void dump_matrix(std::string msg) {
    std::stringstream s;
    s << default_comm().rank() << ": " << msg << " :\n";
    s << default_comm().rank() << ": shape [" << shape_[0] << ", " << shape_[1]
      << " ] seg_size " << segment_size_ << " data_size " << data_size_ << "\n";
    s << default_comm().rank() << ": halo_bounds.prev ";
    for (T *ptr = data_; ptr < data_ + halo_bounds_.prev * shape_[1]; ptr++)
      s << *ptr << " ";
    s << std::endl;
    for (auto r : dm_rows_) {
      if (r.segment()->is_local()) {
        s << default_comm().rank() << ": row " << r.idx() << " : ";
        for (auto _i = rng::begin(r); _i != rng::end(r); ++_i)
          s << *_i << " ";
        s << std::endl;
      }
    }
    s << default_comm().rank() << ": halo_bounds.next ";
    T *_hptr = data_ + halo_bounds_.prev * shape_[1] + segment_size_;
    for (T *ptr = _hptr; ptr < _hptr + halo_bounds_.next * shape_[1]; ptr++)
      s << *ptr << " ";
    s << std::endl << std::endl;
    std::cout << s.str();
  }

  void raw_dump_matrix(std::string msg) {
    std::stringstream s;
    s << default_comm().rank() << ": " << msg << " :\n";
    for (std::size_t i = 0;
         i < segment_shape_[0] + halo_bounds_.prev + halo_bounds_.next; i++) {
      for (std::size_t j = 0; j < segment_shape_[1]; j++) {
        s << data_[i * segment_shape_[1] + j] << " ";
      }
      s << std::endl;
    }
    s << std::endl;
    std::cout << s.str();
  }

  auto data_size() { return data_size_; }

private:
  void init_(dr::halo_bounds hb, auto allocator) {

    halo_bounds_ = hb;
    data_ = allocator.allocate(data_size_);

    grid_size_ = default_comm().size();

    hb.prev *= shape_[1];
    hb.next *= shape_[1];

    halo_ = new dr::span_halo<T>(default_comm(), data_, data_size_, hb);

    // prepare segments
    // one segment per node, 1-d arrangement of segments

    segments_.reserve(grid_size_);

    std::size_t idx = 0;
    for (std::size_t i = 0; i < grid_size_; i++) {
      // T *_ptr = (idx == default_comm().rank()) ? data_ : nullptr;
      // key_type _ts(
      //     segment_shape_[0] -
      //         ((idx + 1) / default_comm().size()) *
      //             (default_comm().size() * segment_shape_[0] - shape_[0]),
      //     segment_shape_[1]);
       std::size_t _seg_rows =
          segment_shape_[0] -
          ((idx + 1) / default_comm().size()) *
              (default_comm().size() * segment_shape_[0] - shape_[0]);   
      // segments_.emplace_back(this, _ptr, _ts, idx);
      segments_.emplace_back(this, idx, _seg_rows * shape_[1]);
      idx++;
    }

    // regular rows
    dm_rows_.reserve(segment_shape_[0]);

    int row_start_index_ = 0;
    for (auto _titr = rng::begin(segments_); _titr != rng::end(segments_);
         ++_titr) {
      for (int _ind = row_start_index_;
           _ind < row_start_index_ + (int)(*_titr).shape()[0]; _ind++) {
        T *_dataptr = nullptr;
        if ((*_titr).is_local()) {
          if (local_rows_indices_.first == -1)
            local_rows_indices_.first = _ind;
          local_rows_indices_.second = _ind;

          int _dataoff = halo_bounds_.prev * segment_shape_[1]; // start of data
          _dataoff += (_ind - default_comm().rank() * segment_shape_[0]) *
                      segment_shape_[1];

          assert(_dataoff >= 0);
          assert(_dataoff < (int)data_size_);
          _dataptr = data_ + _dataoff;
        }
        dm_rows_.emplace_back(_ind, _dataptr, &(*_titr), segment_shape_[1]);
      }
      row_start_index_ += (*_titr).shape()[0];
    };

    // rows in halo.prev area
    for (int _ind = local_rows_indices_.first - halo_bounds_.prev;
         _ind < local_rows_indices_.first; _ind++) {
      std::size_t _dataoff =
          (_ind + halo_bounds_.prev - local_rows_indices_.first) *
          segment_shape_[1];

      assert(_dataoff >= 0);
      assert(_dataoff < halo_bounds_.prev * segment_shape_[1]);
      dm_halop_rows_.emplace_back(_ind, data_ + _dataoff,
                                  &(*rng::begin(segments_)), segment_shape_[1]);
    }

    // rows in halo.next area
    for (int _ind = local_rows_indices_.second + 1;
         _ind < (int)(local_rows_indices_.second + 1 + halo_bounds_.next);
         _ind++) {
      int _dataoff = (_ind + halo_bounds_.prev - local_rows_indices_.first) *
                     segment_shape_[1];

      assert(_dataoff >= 0);
      assert(_dataoff < (int)data_size_);
      dm_halon_rows_.emplace_back(
          _ind, data_ + _dataoff,
          &(*(rng::begin(segments_) + default_comm().rank())),
          segment_shape_[1]);
    }

    win_.create(default_comm(), data_, data_size_ * sizeof(T));
    active_wins().insert(win_.mpi_win());
    dm_segments_ = dm_segments<distributed_dense_matrix>(this);
    fence();
  }

private:
  friend d_segment_iterator<distributed_dense_matrix>;
  friend dm_segments<distributed_dense_matrix>;
  friend dm_rows<distributed_dense_matrix>;
  friend dm_rows_iterator<distributed_dense_matrix>;

  key_type shape_;
  std::size_t grid_size_; // currently (N, 1)
  key_type segment_shape_;

  const std::size_t segment_size_ = 0; // size of local data
  const std::size_t data_size_ = 0;    // all data with halo buffers

  T *data_ = nullptr; // local data ptr

  dr::span_halo<T> *halo_;
  dr::halo_bounds halo_bounds_;
  // std::size_t size_;

  std::vector<d_segment<distributed_dense_matrix>> segments_;
  dm_segments<distributed_dense_matrix>
      dm_segments_; // lightweight view on segments_

  dm_rows<distributed_dense_matrix<T>>
      dm_rows_; // vector of "regular" rows in segment
  dm_rows<distributed_dense_matrix<T>> dm_halop_rows_,
      dm_halon_rows_; // rows in halo area
  std::pair<difference_type, difference_type> local_rows_indices_ =
      std::pair(-1, -1); // global indices of locally stored rows

  dr::rma_window win_;
  Allocator allocator_;
}; // class distributed_dense_matrix

template <typename T>
void for_each(dm_rows<distributed_dense_matrix<T>> &rows, auto op) {
  for (auto itr = rng::begin(rows); itr != rng::end(rows); itr++) {
    if ((*itr).segment()->is_local()) {
      op(*itr);
    }
  }
};

template <typename DM>
void transform(rng::subrange<dm_rows_iterator<DM>> &in,
               dm_rows_iterator<DM> out, auto op) {
  for (auto i = rng::begin(in); i != rng::end(in); i++) {
    if (i.is_local()) {
      *out = op(i);
    }
    ++out;
  }
}

} // namespace dr::mhp

// Needed to satisfy rng::viewable_range
template <typename T>
inline constexpr bool rng::enable_borrowed_range<
    dr::mhp::dm_segments<dr::mhp::distributed_dense_matrix<T>>> = true;

template <typename T>
inline constexpr bool rng::enable_borrowed_range<
    dr::mhp::dm_rows<dr::mhp::distributed_dense_matrix<T>>> = true;

template <typename T>
inline constexpr bool rng::enable_borrowed_range<
    dr::mhp::subrange<dr::mhp::distributed_dense_matrix<T>>> = true;
