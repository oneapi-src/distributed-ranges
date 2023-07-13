// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/mhp/containers/rows.hpp>
#include <dr/mhp/containers/segment.hpp>
#include <dr/mhp/containers/subrange.hpp>

#include <dr/detail/owning_view.hpp>
#ifdef SYCL_LANGUAGE_VERSION
#include <dr/shp/detail.hpp>
#include <dr/shp/views/dense_matrix_view.hpp>
#endif

namespace dr::mhp {

using key_type = dr::index<>;

template <typename T, typename Allocator> class distributed_dense_matrix {
public:
  using value_type = T;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using key_type = index<>;

  distributed_dense_matrix(std::size_t rows, std::size_t cols,
                           distribution dist = distribution(),
                           Allocator allocator = Allocator())
      : distributed_dense_matrix(key_type(rows, cols), dist, allocator){};

  distributed_dense_matrix(std::size_t rows, std::size_t cols, T fillval,
                           distribution dist = distribution(),
                           Allocator allocator = Allocator())
      : distributed_dense_matrix(key_type(rows, cols), dist, allocator) {

    for (std::size_t _i = 0; _i < data_size_; _i++)
      data_[_i] = fillval;
  };

  distributed_dense_matrix(key_type shape, distribution dist = distribution(),
                           Allocator allocator = Allocator())
      : shape_(shape), dm_rows_(this), dm_halo_p_rows_(this),
        dm_halo_n_rows_(this) {
    init_(dist, allocator);
  }
  ~distributed_dense_matrix() {
    fence();
    active_wins().erase(win_.mpi_win());
    win_.free();
    allocator_.deallocate(data_, data_size_);
    data_ = nullptr;
    delete halo_;
  }

  class iterator {
  public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = typename distributed_dense_matrix::value_type;
    using difference_type = typename distributed_dense_matrix::difference_type;

    iterator() {}
    iterator(const distributed_dense_matrix *parent, difference_type offset)
        : parent_(parent), offset_(offset) {}

    auto operator+(difference_type n) const {
      return iterator(parent_, offset_ + n);
    }
    friend auto operator+(difference_type n, const iterator &other) {
      return other + n;
    }
    auto operator-(difference_type n) const {
      return iterator(parent_, offset_ - n);
    }
    auto operator-(iterator other) const { return offset_ - other.offset_; }

    auto &operator+=(difference_type n) {
      offset_ += n;
      return *this;
    }
    auto &operator-=(difference_type n) {
      offset_ -= n;
      return *this;
    }
    auto &operator++() {
      offset_++;
      return *this;
    }
    auto operator++(int) {
      auto old = *this;
      offset_++;
      return old;
    }
    auto &operator--() {
      offset_--;
      return *this;
    }
    auto operator--(int) {
      auto old = *this;
      offset_--;
      return old;
    }

    bool operator==(iterator other) const {
      assert(parent_ == other.parent_);
      return offset_ == other.offset_;
    }
    auto operator<=>(iterator other) const {
      assert(parent_ == other.parent_);
      return offset_ <=> other.offset_;
    }

    auto operator*() const {
      auto segment_size = parent_->segment_size_;
      return parent_
          ->segments()[offset_ / segment_size][offset_ % segment_size];
    }
    auto operator[](difference_type n) const { return *(*this + n); }
    auto operator[](dr::index<difference_type> n) const {
      return *(*this + n[0] * parent_->shape_[1] + n[1]);
    }

    //
    // Support for distributed ranges
    //
    // distributed iterator provides segments
    // remote iterator provides local
    //
    auto segments() {
      return dr::__detail::drop_segments(parent_->segments(), offset_);
    }

  private:
    const distributed_dense_matrix *parent_;
    difference_type offset_;
  };

  T &operator[](difference_type index) {
    assert(index >=
           static_cast<difference_type>(default_comm().rank() * segment_size_));
    assert(index < static_cast<difference_type>((default_comm().rank() + 1) *
                                                segment_size_));
    return *(data_ + distribution_.halo().prev -
             default_comm().rank() * segment_size_ + index);
  }

  T &operator[](dr::index<difference_type> p) {
    assert(p[0] * shape_[1] + p[1] >= default_comm().rank() * segment_size_);
    assert(p[0] * shape_[1] + p[1] <
           (default_comm().rank() + 1) * segment_size_);
    return *(data_ + distribution_.halo().prev -
             default_comm().rank() * segment_size_ + p[0] * shape_[1] + p[1]);
  }

  iterator begin() { return iterator(this, 0); }
  iterator end() { return begin() + size(); }

  dm_rows<distributed_dense_matrix> &rows() { return dm_rows_; }

  T *data() { return data_; }
  auto data_size() { return data_size_; }
  key_type shape() noexcept { return shape_; }
  size_type size() noexcept { return shape()[0] * shape()[1]; }
  auto segments() const { return rng::views::all(segments_); }
  size_type segment_size() const noexcept { return segment_size_; }
  key_type segment_shape() const noexcept { return segment_shape_; }

  auto &halo() {
    assert(halo_ != nullptr);
    return *halo_;
  }
  struct halo_bounds halo_bounds() { return distribution_.halo(); }

#ifdef SYCL_LANGUAGE_VERSION

  // Given a tile index, return a dense matrix view of that tile.
  // dense_matrix_view is a view of a dense tile.
  auto tile(key_type tile_index) { return tile_view_impl_(tile_index); }

  key_type grid_shape() const noexcept {
    return key_type(rng::size(segments_), 1);
  }

  key_type tile_shape() const noexcept { return segment_shape(); }

  auto tile_segments() {
    using tile_type = decltype(tile({0, 0}));
    std::vector<tile_type> tiles;

    for (std::size_t i = 0; i < grid_shape()[0]; i++) {
      for (std::size_t j = 0; j < grid_shape()[1]; j++) {
        auto t =
            tile_view_impl_({i, j}, {i * tile_shape()[0], j * tile_shape()[1]});
        tiles.push_back(t);
      }
    }

    return dr::__detail::owning_view(std::move(tiles));
  }
#endif

  // for debug purposes only
#if 1
  void dump(std::string msg) {
    std::stringstream s;
    s << default_comm().rank() << ": " << msg << " :\n";
    s << default_comm().rank() << ": shape [" << shape_[0] << ", " << shape_[1]
      << " ] seg_size " << segment_size_ << " data_size " << data_size_ << "\n";
    s << default_comm().rank() << ": halo_bounds.prev ";
    for (T *ptr = data_; ptr < data_ + distribution_.halo().prev; ptr++)
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
    T *_hptr = data_ + distribution_.halo().prev + segment_size_;
    for (T *ptr = _hptr; ptr < _hptr + distribution_.halo().next; ptr++)
      s << *ptr << " ";
    s << std::endl << std::endl;
    std::cout << s.str();
  }

  void dump_raw(std::string msg) {
    std::stringstream s;
    s << default_comm().rank() << ": " << msg << " :\n";
    for (std::size_t i = 0; i < segment_shape_[0] + halo_bounds_rows_.prev +
                                    halo_bounds_rows_.next;
         i++) {
      for (std::size_t j = 0; j < shape_[1]; j++) {
        s << data_[i * shape_[1] + j] << " ";
      }
      s << std::endl;
    }
    s << std::endl;
    std::cout << s.str();
  }
#endif

private:
#ifdef SYCL_LANGUAGE_VERSION

  // Return a dense_matrix_view of the tile located at
  // tile grid coordinates tile_index[0], tile_index[1].
  //
  // The row indices of each element in the tile will be incremented
  // by idx_offset[0], and the column indices incremneted by idx_offset[1].
  //
  // When accessing an individual tile, no idx_offset is needed.  (The
  // indices are with respect to the tile itself.)  When viewing a tile as
  // part of the overall matrix (e.g. with segments()), an idx_offset is
  // necessary in order to ensure the correct global indices are observed.
  auto tile_view_impl_(key_type tile_index, key_type idx_offset = {0, 0}) {
    assert(tile_index[1] == 0);

    auto &&segment = segments()[tile_index[1]];

    auto data = rng::begin(segment);

    using Iter = decltype(data);

    auto ld = segment_shape()[1];

    auto tile_shape =
        key_type(std::min(segment_shape()[0],
                          shape()[0] - segment_shape()[0] * tile_index[0]),
                 segment_shape()[1]);

    return dr::shp::dense_matrix_view<T, Iter>(data, tile_shape, idx_offset, ld,
                                               dr::ranges::rank(segment));
  }
#endif

  void init_(distribution dist, auto allocator) {

    auto grid_size_ = default_comm().size(); // dr-style ignore
    assert(shape_[0] > grid_size_);

    segment_shape_ =
        index((shape_[0] + grid_size_ - 1) / grid_size_, shape_[1]);

    //  distribution_ = dist;

    assert(dist.halo().prev <= segment_shape_[0]);
    assert(dist.halo().next <= segment_shape_[0]);
    segment_size_ = segment_shape_[0] * shape_[1];

    distribution_ = distribution().halo(dist.halo().prev * shape_[1],
                                        dist.halo().next * shape_[1]);

    halo_bounds_rows_ = dist.halo();

    data_size_ =
        segment_size_ + distribution_.halo().prev + distribution_.halo().next;

    data_ = allocator.allocate(data_size_);

    halo_ = new span_halo<T>(default_comm(), data_, data_size_,
                             distribution_.halo());

    // prepare sizes and segments
    // one dv_segment per node, 1-d arrangement of segments

    segments_.reserve(grid_size_);

    for (std::size_t idx = 0; idx < grid_size_; idx++) {
      std::size_t _seg_rows =
          (idx + 1) < grid_size_
              ? segment_shape_[0]
              : segment_shape_[0] * (1 - grid_size_) + shape_[0];
      segments_.emplace_back(this, idx, _seg_rows * shape_[1]);
    }

    // regular rows
    dm_rows_.reserve(segment_shape_[0]);

    int row_start_index_ = 0;

    for (auto _sitr = rng::begin(segments_); _sitr != rng::end(segments_);
         ++_sitr) {
      for (int _ind = row_start_index_;
           _ind < row_start_index_ + (int)(rng::distance(*_sitr) / shape_[1]);
           _ind++) {
        T *_dataptr = nullptr;
        if ((*_sitr).is_local()) {
          if (local_rows_ind_.first == -1)
            local_rows_ind_.first = _ind;
          local_rows_ind_.second = _ind;

          int _dataoff = distribution_.halo().prev; // start of data
          _dataoff +=
              (_ind - default_comm().rank() * segment_shape_[0]) * shape_[1];

          assert(_dataoff >= 0);
          assert(_dataoff < (int)data_size_);
          _dataptr = data_ + _dataoff;
        }
        dm_rows_.emplace_back(_ind, _dataptr, shape_[1], &(*_sitr));
      }
      row_start_index_ += rng::distance(*_sitr) / shape_[1];
    };
    // barrier();

    // rows in halo.prev area
    for (int _ind = local_rows_ind_.first - halo_bounds_rows_.prev;
         _ind < local_rows_ind_.first; _ind++) {
      std::size_t _dataoff = distribution_.halo().prev +
                             (_ind - local_rows_ind_.first) * shape_[1];

      assert(_dataoff >= 0);
      assert(_dataoff < distribution_.halo().prev);
      dm_halo_p_rows_.emplace_back(_ind, data_ + _dataoff, shape_[1],
                                   &(*rng::begin(segments_)));
    }

    // rows in halo.next area
    for (int _ind = local_rows_ind_.second + 1;
         _ind < (int)(local_rows_ind_.second + 1 + halo_bounds_rows_.next);
         _ind++) {
      int _dataoff = distribution_.halo().prev +
                     (_ind - local_rows_ind_.first) * shape_[1];

      assert(_dataoff >= 0);
      assert(_dataoff < (int)data_size_);
      dm_halo_n_rows_.emplace_back(
          _ind, data_ + _dataoff, shape_[1],
          &(*(rng::begin(segments_) + default_comm().rank())));
    }

    win_.create(default_comm(), data_, data_size_ * sizeof(T));
    active_wins().insert(win_.mpi_win());
    fence();
  }

  bool is_local_row(int index) {
    if (index >= local_rows_ind_.first && index <= local_rows_ind_.second) {
      return true;
    } else {
      return false;
    }
  }
  // index of cell on linear view
  bool is_local_cell(int index) {
    if (index >= local_rows_ind_.first * (int)shape_[1] &&
        index < (local_rows_ind_.second + 1) * (int)shape_[1]) {
      return true;
    } else {
      return false;
    }
  }

private:
  friend dv_segment_iterator<distributed_dense_matrix>;
  friend dm_rows<distributed_dense_matrix>;
  friend dm_rows_iterator<distributed_dense_matrix>;
  friend subrange_iterator<distributed_dense_matrix>;

  key_type shape_; // matrix shape
  key_type segment_shape_;

  std::size_t segment_size_ = 0; // size of local data
  std::size_t data_size_ = 0;    // all data - local + halo

  T *data_ = nullptr; // data ptr

  span_halo<T> *halo_ = nullptr;
  // halo boundaries counted in cells and rows
  struct halo_bounds halo_bounds_rows_;
  distribution distribution_;

  std::vector<dv_segment<distributed_dense_matrix>> segments_;

  // vector of "regular" rows in segment
  dm_rows<distributed_dense_matrix> dm_rows_;
  // rows in halo areas, prev & next
  dm_rows<distributed_dense_matrix> dm_halo_p_rows_, dm_halo_n_rows_;

  // global indices of locally stored rows (lowest and highest)
  std::pair<difference_type, difference_type> local_rows_ind_ =
      std::pair(-1, -1);

  dr::rma_window win_;
  Allocator allocator_;
}; // class distributed_dense_matrix

//
// for_each, but iterator is passed to lambda
// FIXME: enable sycl
//
template <typename T>
auto for_each(dm_rows<distributed_dense_matrix<T>> &rows, auto op) {
  for (auto itr = rng::begin(rows); itr != rng::end(rows); itr++) {
    if ((*itr).segment()->is_local()) {
      op(*itr);
    }
  }
};

#if 0 // temporary disabled, to be fixed
template <typename T>
concept has_segment = requires(T t) { (*(t.begin())).segment(); };

template <typename T1, typename T2>
  requires(has_segment<T1> && has_segment<T2>)
auto for_each(zip_view<T1, T2> &v, auto op) {
  for (auto itr = rng::begin(v); itr != rng::end(v); itr++) {
    auto [in, out] = *itr;
    if (in.segment()->is_local()) {
      assert(out.segment()->is_local());
      op(*itr);
    }
  }
};
#endif
//
// transform, iterator pointing at element is passed to lambda
//
template <typename DM>
void transform(dr::mhp::subrange<DM> &in, subrange_iterator<DM> out, auto op) {
  for (subrange_iterator<DM> i = rng::begin(in); i != rng::end(in); i++) {
    if (i.is_local()) {
      assert(out.is_local());
      *(out) = op(i);
    }
    ++out;
  }
}

template <typename DM>
void transform(rng::subrange<dm_rows_iterator<DM>> &in,
               dm_rows_iterator<DM> out, auto op) {
  for (auto i = rng::begin(in); i != rng::end(in); i++) {
    if (i.is_local()) {
      assert(out.is_local());
      *out = op(i);
    }
    ++out;
  }
}

} // namespace dr::mhp

// Needed to satisfy rng::viewable_range

template <typename T>
inline constexpr bool rng::enable_borrowed_range<
    dr::mhp::dm_rows<dr::mhp::distributed_dense_matrix<T>>> = true;
