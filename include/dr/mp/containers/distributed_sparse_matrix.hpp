// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <dr/mp/containers/matrix_formats/csr_eq_distribution.hpp>
#include <dr/mp/containers/matrix_formats/csr_row_distribution.hpp>
#include<dr/detail/matrix_entry.hpp>
#include<dr/views/csr_matrix_view.hpp>


namespace dr::mp {
template <typename T>
concept matrix_distibution =
    requires(T t) {
      {t.fence()} -> std::same_as<void>;
      { t.segments() } -> rng::random_access_range;
      {t.shape().first} -> std::convertible_to<std::size_t>;
      {t.shape().second} -> std::convertible_to<std::size_t>;
      {t.nnz()} -> std::same_as<std::size_t>;
      {t.get_segment_from_offset(int())} -> std::same_as<std::size_t>;
      {t.get_id_in_segment(int())} -> std::same_as<std::size_t>;
      T(dr::views::csr_matrix_view<typename T::elem_type, typename T::index_type>(), distribution());
    };

template <typename T, typename I, class BackendT = MpiBackend, class MatrixDistrT = csr_row_distribution<T, I, BackendT>>
requires(matrix_distibution<MatrixDistrT>)
class distributed_sparse_matrix {

public:
  using value_type = dr::matrix_entry<T, I>;
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
      auto segment_id = parent_->distribution_.get_segment_from_offset(offset_);
      auto id_in_segment = parent_->distribution_.get_id_in_segment(offset_);
      return parent_->segments()[segment_id][id_in_segment];
    }
    auto operator[](difference_type n) const { return *(*this + n); }

    auto local() {
      auto segment_id = parent_->distribution_.get_segment_from_offset(offset_);
      auto id_in_segment = parent_->distribution_.get_id_in_segment(offset_);
      return (parent_->segments()[segment_id].begin() + id_in_segment).local();
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
  distributed_sparse_matrix(dr::views::csr_matrix_view<T, I> csr_view, distribution dist = distribution()): distribution_(csr_view, dist) {}

  /// Returns iterator to beginning
  auto begin() const { return iterator(this, 0); }
  /// Returns iterator to end
  auto end() const { return begin() + distribution_.nnz(); }

  /// Returns size
  auto size() const { return distribution_.nnz(); }

  auto shape() const { return distribution_.shape(); }
  /// Returns reference using index
  auto operator[](difference_type n) const { return *(begin() + n); }
//   auto &halo() const { return *halo_; } TODO

  auto segments() const { return distribution_.segments(); }

  void fence() { 
    distribution_.fence();
   }

  template<typename C, typename A>
  auto local_gemv_and_collect(std::size_t root, C &res, A &vals) const {
    distribution_.local_gemv_and_collect(root, res, vals);
  }

private:
  MatrixDistrT distribution_;

};
}