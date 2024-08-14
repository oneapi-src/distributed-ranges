// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <dr/mp/containers/sparse_matrix_segment.hpp>
#include<dr/sp/containers/matrix_entry.hpp>
#include<dr/sp/views/csr_matrix_view.hpp>


namespace dr::mp {


template <typename T, typename I, class BackendT = MpiBackend> class distributed_sparse_matrix {

public:
  using value_type = dr::sp::matrix_entry<T, I>;
  using elem_type = T;
  using index_type = I;
  using key_type = dr::index<I>;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using backend_type = BackendT;

  class iterator {
  public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = typename distributed_sparse_matrix::value_type;
    using difference_type = typename distributed_sparse_matrix::difference_type;

    iterator() {}
    iterator(const distributed_sparse_matrix *parent, difference_type offset)
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
      if (parent_ == nullptr || other.parent_ == nullptr) {
        return false;
      } else {
        return offset_ == other.offset_;
      }
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

    auto local() {
      auto segment_size = parent_->segment_size_;
      return (parent_->segments()[offset_ / segment_size].begin() +
              offset_ % segment_size)
          .local();
    }

    auto segments() {
      return dr::__detail::drop_segments(parent_->segments(), offset_);
    }

  private:
    const distributed_sparse_matrix *parent_ = nullptr;
    difference_type offset_;
  };

  distributed_sparse_matrix(const distributed_sparse_matrix &) = delete;
  distributed_sparse_matrix &operator=(const distributed_sparse_matrix &) = delete;
  distributed_sparse_matrix(distributed_sparse_matrix &&) { assert(false); }

  /// Constructor
  distributed_sparse_matrix(dr::sp::csr_matrix_view<T, I> csr_view, distribution dist = distribution()) {
    init(csr_view, dist);
  }

  ~distributed_sparse_matrix() {
    if (!finalized()) {
      fence();
      if (rows_data_ != nullptr) {
        rows_backend_.deallocate(rows_data_, row_size_ * sizeof(index_type));
      }

    //   delete halo_; TODO
    }
  }

  /// Returns iterator to beginning
  auto begin() const { return iterator(this, 0); }
  /// Returns iterator to end
  auto end() const { return begin() + nnz_; }

  /// Returns size
  auto size() const { return nnz_; }

  auto shape() const { return shape_; }
  /// Returns reference using index
  auto operator[](difference_type n) const { return *(begin() + n); }
//   auto &halo() const { return *halo_; } TODO

  auto segments() const { return rng::views::all(segments_); }

  void fence() { 
    rows_backend_.fence(); // it does not matter which backend we choose, since all of them share comm
   }

private:

  friend dsm_segment_iterator<distributed_sparse_matrix>;
  std::size_t get_row_size(std::size_t rank) {
    std::size_t start_index = row_offsets_[rank];
    std::size_t end_index = nnz_;
    if (rank + 1 < row_offsets_.size()) {
      end_index = row_offsets_[rank + 1];
    }
    return end_index - start_index;
  }

  void init(dr::sp::csr_matrix_view<T, I> csr_view, auto dist) {
    nnz_ = csr_view.size();
    distribution_ = dist;
    shape_ = csr_view.shape();
    // determine the distribution of data
    // auto hb = dist.halo();
    std::size_t gran = dist.granularity();
    // TODO: make this an error that is reported back to user
    assert(nnz_ % gran == 0 && "size must be a multiple of the granularity");
    // assert(hb.prev % gran == 0 && "size must be a multiple of the granularity");
    // assert(hb.next % gran == 0 && "size must be a multiple of the granularity");


    auto rank = rows_backend_.getrank();
    vals_data_ = std::make_shared<distributed_vector<T>>(nnz_);
    cols_data_ = std::make_shared<distributed_vector<I>>(nnz_);

    dr::mp::copy(std::ranges::subrange(csr_view.values_data(), csr_view.values_data() + nnz_), vals_data_->begin());
    dr::mp::copy(std::ranges::subrange(csr_view.colind_data(), csr_view.colind_data() + nnz_), cols_data_->begin());
    
    assert(*csr_view.rowptr_data() == 0);
    for (int i = 0; i < default_comm().size(); i++) {
      auto first_index = vals_data_->get_segment_offset(i);
      auto lower_limit = std::distance(csr_view.rowptr_data(), std::upper_bound(csr_view.rowptr_data(), csr_view.rowptr_data() + shape_[0], first_index)) - 1;
      row_offsets_.push_back(lower_limit);
    }

    auto last_index = vals_data_->get_segment_offset(rank + 1) - 1;

    auto lower_limit = row_offsets_[rank];
    auto higher_limit = std::distance(csr_view.rowptr_data(), std::upper_bound(csr_view.rowptr_data(), csr_view.rowptr_data() + shape_[0], last_index));
    row_size_ = higher_limit - lower_limit;
    
    rows_data_ = static_cast<I *>(rows_backend_.allocate(row_size_ * sizeof(I)));
    std::copy(csr_view.rowptr_data() + lower_limit, csr_view.rowptr_data() + higher_limit, rows_data_);
    std::size_t segment_index = 0;
    segment_size_ = vals_data_->segment_size();
    assert(segment_size_ == cols_data_->segment_size());
    for (std::size_t i = 0; i < nnz_; i += segment_size_) {
      segments_.emplace_back(this, segment_index++,
                             std::min(segment_size_, nnz_ - i), segment_size_);
    }
      
    fence();
  }


  std::size_t segment_size_ = 0;
  std::size_t row_size_ = 0;
  std::vector<std::size_t> row_offsets_;

  index_type *rows_data_ = nullptr;
  BackendT rows_backend_;

  distribution distribution_;
  dr::index<I> shape_;
  std::size_t nnz_;
  std::vector<dsm_segment<distributed_sparse_matrix>> segments_;
  std::shared_ptr<distributed_vector<T>> vals_data_;
  std::shared_ptr<distributed_vector<I>> cols_data_;
};
}