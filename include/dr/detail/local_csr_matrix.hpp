// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/detail/matrix_entry.hpp>
#include <memory>
#include <vector>
#include <future>

namespace dr {

namespace __detail {

template <typename T, typename I, typename Allocator = std::allocator<T>>
class local_csr_matrix {
public:
  using value_type = std::pair<T, I>;
  using scalar_type = T;
  using index_type = I;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  using allocator_type = Allocator;

  using key_type = dr::index<I>;
  using map_type = T;

  using backend_allocator_type = typename std::allocator_traits<
      allocator_type>::template rebind_alloc<value_type>;
  using aggregator_allocator_type = typename std::allocator_traits<
      allocator_type>::template rebind_alloc<std::vector<value_type>>;
  using row_type = std::vector<value_type, backend_allocator_type>;
  using backend_type = std::vector<row_type, aggregator_allocator_type>;

  using iterator = typename backend_type::iterator;
  using const_iterator = typename backend_type::const_iterator;

  local_csr_matrix(dr::index<I> shape, std::size_t nnz) : shape_(shape) {
    auto average_size = nnz / shape.first / 2;
    for (std::size_t i = 0; i < shape.first; i++) {
      tuples_.push_back(row_type());
      tuples_.back().reserve(average_size);
    }
  }

  dr::index<I> shape() const noexcept { return shape_; }

  size_type size() const noexcept { return size_; }

  iterator begin() noexcept { return tuples_.begin(); }

  const_iterator begin() const noexcept { return tuples_.begin(); }

  iterator end() noexcept { return tuples_.end(); }

  const_iterator end() const noexcept { return tuples_.end(); }

  template <typename InputIt> void push_back(InputIt first, InputIt last) {
    for (auto iter = first; iter != last; ++iter) {
      push_back(*iter);
    }
  }

  void push_back(index_type row, const value_type &value) { 
    tuples_[row].push_back(value); 
    size_++;
  }


  void sort() {
    auto comparator = [](auto &one, auto& two) {
      return one.second < two.second;
    };
    for (auto &elem: tuples_) {
      std::sort(elem.begin(), elem.end(), comparator);
    }
  }

  local_csr_matrix() = default;
  ~local_csr_matrix() = default;
  local_csr_matrix(const local_csr_matrix &) = default;
  local_csr_matrix(local_csr_matrix &&) = default;
  local_csr_matrix &operator=(const local_csr_matrix &) = default;
  local_csr_matrix &operator=(local_csr_matrix &&) = default;

private:
  std::size_t size_ = 0;
  dr::index<I> shape_;
  backend_type tuples_;
};

} // namespace __detail

} // namespace dr
