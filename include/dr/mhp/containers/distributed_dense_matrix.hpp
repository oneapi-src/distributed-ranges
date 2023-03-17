// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "index.hpp"

namespace mhp {

template <typename T> class dm_tile {
public:
  using key_type = mhp::index<>;

  dm_tile(T * data, key_type shape, size_t rank) 
    : data_(data), rank_(rank), shape_(shape) {};

  bool is_local() { return rank_ == default_comm().rank(); }

  size_t rank() { return rank_; }
  key_type shape() { return shape_; }
private:
  T * data_ = nullptr;
  size_t rank_;
  key_type shape_;

};

template <typename T> class dm_row_iterator : 
  public std::iterator<std::forward_iterator_tag, T> {
public:
  dm_row_iterator(T* p, dm_tile<T> & tile) : p_(p), tile_(tile) {};

  dm_row_iterator& operator++() { p_++; return *this; };
  bool operator==(dm_row_iterator other) const { return p_ == other.p_; }
  bool operator!=(dm_row_iterator other) const { return p_ != other.p_; }
  T & operator*() { return *p_; }

private:
  T* p_;
  dm_tile<T> & tile_;
};

template <typename T> class dm_row_view {
  using iterator = dm_row_iterator<T>;
public:
  dm_row_view(T * ptr, dm_tile<T> & tile, size_t size) : data_(ptr), tileref_(tile), size_(size) {};

  iterator begin() { return  iterator(data_, tileref_); }
  iterator end() { return  iterator(data_ + size_, tileref_); }
private:
  T * data_ = nullptr;
  dm_tile<T> & tileref_;
  size_t size_ = 0;
};

template <typename T, typename Allocator = std::allocator<T>> 
class distributed_dense_matrix {
public:
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  // using value_type = mhp::matrix_entry<T>;

  using key_type = mhp::index<>;

  using iterator =
      lib::iterator_adaptor<distributed_dense_matrix<T>>;

  size_type size() const noexcept { return shape()[0] * shape()[1]; }

  key_type shape() const noexcept { return shape_; }

    
  std::vector<dm_row_view<T>> local_rows() {
    std::vector<dm_row_view<T>> row_view_;

    for(auto _titr = tiles_.begin(); _titr != tiles_.end(); ++_titr) {
      if ((*_titr).is_local()) {
        for (size_t s = 0; s < (*_titr).shape()[0]; s++) {
          row_view_.emplace_back(data_ + s * (*_titr).shape()[1], *_titr, (*_titr).shape()[1]);
        }
      }
    };
    return row_view_;
  }

  std::vector<dm_row_view<T>> rows() {
    std::vector<dm_row_view<T>> row_view_;

    for(auto _titr = tiles_.begin(); _titr != tiles_.end(); ++_titr) {
      for (size_t s = 0; s < (*_titr).shape()[0]; s++) {
        T * _dataptr = (*_titr).is_local() ? (data_ + s * (*_titr).shape()[1]) : nullptr;
        row_view_.emplace_back(_dataptr, *_titr, (*_titr).shape()[1]);
      }
    };
    return row_view_;
  }



  distributed_dense_matrix(key_type shape, lib::halo_bounds hb = lib::halo_bounds(), Allocator allocator = Allocator())
      : shape_(shape), // partition_(new mhp::block_cyclic()),
      grid_shape_(default_comm().size(), 1),
      tile_shape_(( shape_[0] + default_comm().size() - 1)/default_comm().size(), shape_[1]),
      segment_size_( std::max({ tile_shape_[0] * tile_shape_[1], hb.prev * tile_shape_[1], hb.next * tile_shape_[1] } )),
      data_size_(segment_size_ + hb.prev * shape_[1] + hb.next * shape_[1]),
      data_(allocator.allocate(data_size_)),
      halo_(lib::span_halo<T>(default_comm(), data_, data_size_, hb))
 {
    init_();
  }

  distributed_dense_matrix(size_t rows, size_t cols, lib::halo_bounds hb = lib::halo_bounds(), Allocator allocator = Allocator())
      : shape_(key_type(rows, cols)), // partition_(new mhp::block_cyclic()), 
      grid_shape_(default_comm().size(), 1),
      tile_shape_(( shape_[0] + default_comm().size() - 1)/default_comm().size(), shape_[1]),
      segment_size_( std::max({ tile_shape_[0] * tile_shape_[1], hb.prev * tile_shape_[1], hb.next * tile_shape_[1] } )),
      data_size_(segment_size_ + hb.prev * shape_[1] + hb.next * shape_[1]),
      data_(allocator.allocate(data_size_)),
      halo_(lib::span_halo<T>(default_comm(), data_, data_size_, hb))
  {
    init_();
  }

private:
  void init_() {

    // one tile per node, 1-d arrangement of tiles

    tiles_.reserve(grid_shape_[0] * grid_shape_[1]);

    size_t idx = 0;
    for (std::size_t i = 0; i < grid_shape_[0]; i++) {
      for (std::size_t j = 0; j < grid_shape_[1]; j++) {
        T * _ptr = (idx == default_comm().rank())?data_:nullptr;

        key_type _ts( tile_shape_[0] - ((idx + 1) / default_comm().size()) * (default_comm().size() * tile_shape_[0] - shape_[0]), 
                      tile_shape_[1]);
        tiles_.emplace_back(_ptr, _ts, idx);
        idx++;
      }
    }
  }

private:
  key_type shape_;
  key_type grid_shape_;  // currently (N, 1)
  key_type tile_shape_;

  std::size_t segment_size_ = 0;
  std::size_t data_size_ = 0;
  T *data_ = nullptr;
  lib::span_halo<T> halo_;
  lib::halo_bounds halo_bounds_;
  std::size_t size_;

  // std::vector<dm_segment<distributed_dense_matrix>> segments_;
  // dm_segments<distributed_dense_matrix> dm_segments_;

  std::vector<dm_tile<T>> tiles_;
  lib::rma_window win_;
  Allocator allocator_;
};

template<typename T>
void for_each(dm_row_iterator<T> First, dm_row_iterator<T> Last, auto op) {
  for (auto itr = First; itr != Last; itr++) {
    if ((*itr).tile().is_local())
      op(*itr);
  }
};

} // namespace mhp
