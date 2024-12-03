// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <dr/detail/matrix_entry.hpp>
#include <dr/detail/multiply_view.hpp>
#include <dr/mp/containers/matrix_formats/csr_eq_segment.hpp>
#include <dr/views/csr_matrix_view.hpp>

namespace dr::mp {

template <typename T, typename I, class BackendT = MpiBackend>
class csr_eq_distribution {
  using view_tuple = std::tuple<std::size_t, std::size_t, std::size_t, I *>;

public:
  using value_type = dr::matrix_entry<T, I>;
  using segment_type = csr_eq_segment<csr_eq_distribution>;
  using elem_type = T;
  using index_type = I;
  using difference_type = std::ptrdiff_t;

  csr_eq_distribution(const csr_eq_distribution &) = delete;
  csr_eq_distribution &operator=(const csr_eq_distribution &) = delete;
  csr_eq_distribution(csr_eq_distribution &&) { assert(false); }

  csr_eq_distribution(dr::views::csr_matrix_view<T, I> csr_view,
                      distribution dist = distribution(),
                      std::size_t root = 0) {
    init(csr_view, dist, root);
  }

  ~csr_eq_distribution() {
    if (!finalized()) {
      fence();
      if (rows_data_ != nullptr) {
        rows_backend_.deallocate(rows_data_, row_size_ * sizeof(index_type));
        tuple_alloc.deallocate(view_helper_const, 1);
      }
    }
  }
  std::size_t get_id_in_segment(std::size_t offset) const {
    return offset % segment_size_;
  }
  std::size_t get_segment_from_offset(std::size_t offset) const {
    return offset / segment_size_;
  }
  auto segments() const { return rng::views::all(segments_); }
  auto nnz() const { return nnz_; }
  auto shape() const { return shape_; }
  void fence() const { rows_backend_.fence(); }

  template <typename C>
  auto local_gemv(C &res, T *vals, std::size_t vals_width) const {
    auto rank = rows_backend_.getrank();
    if (nnz_ <= segment_size_ * rank) {
      return;
    }
    auto vals_len = shape_[1];
    auto size = row_sizes_[rank];
    auto res_col_len = row_sizes_[default_comm().rank()];
    if (dr::mp::use_sycl()) {
      auto localVals = dr::__detail::direct_iterator(
          dr::mp::local_segment(*vals_data_).begin());
      auto localCols = dr::__detail::direct_iterator(
          dr::mp::local_segment(*cols_data_).begin());
      auto offset = rank * segment_size_;
      auto real_segment_size =
          std::min(nnz_ - rank * segment_size_, segment_size_);
      auto local_data = rows_data_;
      auto division = std::max(1ul, real_segment_size / 50);
      auto one_computation_size = (real_segment_size + division - 1) / division;
      auto row_size = row_size_;
      dr::__detail::parallel_for_workaround(
          dr::mp::sycl_queue(), sycl::range<1>{division},
          [=](auto idx) {
            std::size_t lower_bound = one_computation_size * idx;
            std::size_t upper_bound =
                std::min(one_computation_size * (idx + 1), real_segment_size);
            std::size_t position = lower_bound + offset;
            std::size_t first_row = rng::distance(
                local_data,
                std::upper_bound(local_data, local_data + row_size, position) -
                    1);
            for (auto j = 0; j < vals_width; j++) {
              auto row = first_row;
              T sum = 0;

              for (auto i = lower_bound; i < upper_bound; i++) {
                while (row + 1 < row_size &&
                       local_data[row + 1] <= offset + i) {
                  sycl::atomic_ref<T, sycl::memory_order::relaxed,
                                   sycl::memory_scope::device>
                      c_ref(res[row + j * res_col_len]);
                  c_ref += sum;
                  row++;
                  sum = 0;
                }
                auto colNum = localCols[i] + j * vals_len;
                auto matrixVal = vals[colNum];
                auto vectorVal = localVals[i];

                sum += matrixVal * vectorVal;
              }
              sycl::atomic_ref<T, sycl::memory_order::relaxed,
                               sycl::memory_scope::device>
                  c_ref(res[row + j * res_col_len]);
              c_ref += sum;
            }
          })
          .wait();
    } else {
      auto row_i = -1;
      auto position = segment_size_ * rank;
      auto elem_count = std::min(segment_size_, nnz_ - segment_size_ * rank);
      auto current_row_position = rows_data_[0];
      auto local_vals = dr::mp::local_segment(*vals_data_);
      auto local_cols = dr::mp::local_segment(*cols_data_);

      for (int i = 0; i < elem_count; i++) {
        while (row_i + 1 < size && position + i >= current_row_position) {
          row_i++;
          current_row_position = rows_data_[row_i + 1];
        }
        for (int j = 0; j < vals_width; j++) {
          res[row_i + j * res_col_len] +=
              local_vals[i] * vals[local_cols[i] + j * vals_len];
        }
      }
    }
  }

  template <typename C>
  auto local_gemv_and_collect(std::size_t root, C &res, T *vals,
                              std::size_t vals_width) const {
    assert(res.size() == shape_.first * vals_width);
    __detail::allocator<T> alloc;
    auto res_alloc =
        alloc.allocate(row_sizes_[default_comm().rank()] * vals_width);
    if (use_sycl()) {
      sycl_queue()
          .fill(res_alloc, 0, row_sizes_[default_comm().rank()] * vals_width)
          .wait();
    } else {
      std::fill(res_alloc,
                res_alloc + row_sizes_[default_comm().rank()] * vals_width, 0);
    }

    local_gemv(res_alloc, vals, vals_width);
    gather_gemv_vector(root, res, res_alloc, vals_width);
    fence();
    alloc.deallocate(res_alloc, row_sizes_[default_comm().rank()] * vals_width);
  }

private:
  friend csr_eq_segment_iterator<csr_eq_distribution>;

  template <typename C, typename A>
  void gather_gemv_vector(std::size_t root, C &res, A &partial_res,
                          std::size_t vals_width) const {
    auto communicator = default_comm();
    __detail::allocator<T> alloc;
    long long *counts = new long long[communicator.size()];
    for (auto i = 0; i < communicator.size(); i++) {
      counts[i] = row_sizes_[i] * sizeof(T) * vals_width;
    }

    if (communicator.rank() == root) {
      long *offsets = new long[communicator.size()];
      offsets[0] = 0;
      for (auto i = 0; i < communicator.size() - 1; i++) {
        offsets[i + 1] = offsets[i] + counts[i];
      }
      auto gathered_res = alloc.allocate(max_row_size_ * vals_width);
      communicator.gatherv(partial_res, counts, offsets, gathered_res, root);
      T *gathered_res_host;

      if (use_sycl()) {
        gathered_res_host = new T[max_row_size_ * vals_width];
        __detail::sycl_copy(gathered_res, gathered_res_host,
                            max_row_size_ * vals_width);
      } else {
        gathered_res_host = gathered_res;
      }
      rng::fill(res, 0);

      for (auto k = 0; k < vals_width; k++) {
        auto current_offset = 0;
        for (auto i = 0; i < communicator.size(); i++) {
          auto first_row = row_offsets_[i];
          auto last_row = row_offsets_[i] + row_sizes_[i];
          auto row_size = row_sizes_[i];
          if (first_row < last_row) {
            res[first_row + k * shape_[0]] +=
                gathered_res_host[vals_width * current_offset + k * row_size];
          }
          if (first_row < last_row - 1) {
            auto piece_start = gathered_res_host + vals_width * current_offset +
                               k * row_size + 1;
            std::copy(piece_start, piece_start + last_row - first_row - 1,
                      res.begin() + first_row + k * shape_[0] + 1);
          }
          // for (auto j = first_row; j < last_row; j++) {
          //   res[j + k * shape_[0]] +=
          //       gathered_res_host[vals_width * current_offset + k * row_size
          //       +
          //                         j - first_row];
          // }
          current_offset += row_sizes_[i];
        }
      }

      if (use_sycl()) {
        delete[] gathered_res_host;
      }
      delete[] offsets;
      alloc.deallocate(gathered_res, max_row_size_ * vals_width);
    } else {
      communicator.gatherv(partial_res, counts, nullptr, nullptr, root);
    }
    delete[] counts;
  }

  std::size_t get_row_size(std::size_t rank) { return row_sizes_[rank]; }

  void init(dr::views::csr_matrix_view<T, I> csr_view, auto dist,
            std::size_t root) {
    distribution_ = dist;
    auto rank = rows_backend_.getrank();

    std::size_t initial_data[3];
    if (root == rank) {
      initial_data[0] = csr_view.size();
      initial_data[1] = csr_view.shape().first;
      initial_data[2] = csr_view.shape().second;
      default_comm().bcast(initial_data, sizeof(std::size_t) * 3, root);
    } else {
      default_comm().bcast(initial_data, sizeof(std::size_t) * 3, root);
    }

    nnz_ = initial_data[0];
    shape_ = {initial_data[1], initial_data[2]};
    vals_data_ = std::make_shared<distributed_vector<T>>(nnz_);
    cols_data_ = std::make_shared<distributed_vector<I>>(nnz_);
    dr::mp::copy(root,
                 std::ranges::subrange(csr_view.values_data(),
                                       csr_view.values_data() + nnz_),
                 vals_data_->begin());
    dr::mp::copy(root,
                 std::ranges::subrange(csr_view.colind_data(),
                                       csr_view.colind_data() + nnz_),
                 cols_data_->begin());

    auto row_info_size = default_comm().size() * 2 + 1;
    __detail::allocator<size_t> alloc;
    std::size_t *row_information = new std::size_t[row_info_size];
    row_offsets_.reserve(default_comm().size());
    row_sizes_.reserve(default_comm().size());
    if (root == default_comm().rank()) {
      for (int i = 0; i < default_comm().size(); i++) {
        auto first_index = vals_data_->get_segment_offset(i);
        auto last_index = vals_data_->get_segment_offset(i + 1) - 1;
        auto lower_limit =
            rng::distance(csr_view.rowptr_data(),
                          std::upper_bound(csr_view.rowptr_data(),
                                           csr_view.rowptr_data() + shape_[0],
                                           first_index)) -
            1;
        auto higher_limit = rng::distance(
            csr_view.rowptr_data(),
            std::upper_bound(csr_view.rowptr_data(),
                             csr_view.rowptr_data() + shape_[0], last_index));
        row_offsets_.push_back(lower_limit);
        row_sizes_.push_back(higher_limit - lower_limit);
        row_information[i] = lower_limit;
        row_information[default_comm().size() + i] = higher_limit - lower_limit;
        max_row_size_ = max_row_size_ + row_sizes_.back();
      }
      row_information[default_comm().size() * 2] = max_row_size_;
      default_comm().bcast(row_information, sizeof(std::size_t) * row_info_size,
                           root);
    } else {
      default_comm().bcast(row_information, sizeof(std::size_t) * row_info_size,
                           root);
      for (int i = 0; i < default_comm().size(); i++) {
        row_offsets_.push_back(row_information[i]);
        row_sizes_.push_back(row_information[default_comm().size() + i]);
      }
      max_row_size_ = row_information[default_comm().size() * 2];
    }
    delete[] row_information;
    row_size_ = std::max(row_sizes_[rank], static_cast<std::size_t>(1));
    rows_data_ =
        static_cast<I *>(rows_backend_.allocate(row_size_ * sizeof(I)));

    fence();
    if (rank == root) {
      for (std::size_t i = 0; i < default_comm().size(); i++) {
        auto lower_limit = row_offsets_[i];
        auto row_size = row_sizes_[i];
        if (row_size > 0) {
          rows_backend_.putmem(csr_view.rowptr_data() + lower_limit, 0,
                               row_size * sizeof(I), i);
        }
      }
    }

    std::size_t segment_index = 0;
    segment_size_ = vals_data_->segment_size();
    assert(segment_size_ == cols_data_->segment_size());
    for (std::size_t i = 0; i < nnz_; i += segment_size_) {
      segments_.emplace_back(this, segment_index++,
                             std::min(segment_size_, nnz_ - i), segment_size_);
    }
    auto local_rows = rows_data_;
    auto real_val_size = std::min(vals_data_->segment_size(),
                                  nnz_ - vals_data_->segment_size() * rank);
    auto my_tuple = std::make_tuple(row_size_, row_offsets_[rank],
                                    segment_size_ * rank, local_rows);
    view_helper_const = tuple_alloc.allocate(1);

    if (use_sycl()) {
      sycl_queue()
          .memcpy(view_helper_const, &my_tuple, sizeof(view_tuple))
          .wait();
    } else {
      view_helper_const[0] = my_tuple;
    }

    auto local_cols = static_cast<I *>(nullptr);
    auto local_vals = static_cast<T *>(nullptr);
    if (cols_data_->segments().size() > rank) {
      local_cols = cols_data_->segments()[rank].begin().local();
      local_vals = vals_data_->segments()[rank].begin().local();
      local_view = std::make_shared<view_type>(get_elem_view(
          real_val_size, view_helper_const, local_cols, local_vals, rank));
    }
    fence();
  }

  static auto get_elem_view(std::size_t vals_size, view_tuple *helper_tuple,
                            index_type *local_cols, elem_type *local_vals,
                            std::size_t rank) {
    auto local_vals_range = rng::subrange(local_vals, local_vals + vals_size);
    auto local_cols_range = rng::subrange(local_cols, local_cols + vals_size);
    auto zipped_results = rng::views::zip(local_vals_range, local_cols_range);
    auto enumerated_zipped = rng::views::enumerate(zipped_results);
    // we need to use multiply_view here,
    // because lambda is not properly copied to sycl environment
    // when we use variable capture
    auto multiply_range = dr::__detail::multiply_view(
        rng::subrange(helper_tuple, helper_tuple + 1), vals_size);
    auto enumerted_with_data =
        rng::views::zip(enumerated_zipped, multiply_range);

    auto transformer = [=](auto x) {
      auto [entry, tuple] = x;
      auto [row_size, row_offset, offset, local_rows] = tuple;
      auto [index, pair] = entry;
      auto [val, column] = pair;
      auto row =
          rng::distance(local_rows,
                        std::upper_bound(local_rows, local_rows + row_size,
                                         offset + index) -
                            1) +
          row_offset;
      dr::index<index_type> index_obj(row, column);
      value_type entry_obj(index_obj, val);
      return entry_obj;
    };
    return rng::transform_view(enumerted_with_data, std::move(transformer));
  }

  using view_type = decltype(get_elem_view(0, nullptr, nullptr, nullptr, 0));

  dr::mp::__detail::allocator<view_tuple> tuple_alloc;
  view_tuple *view_helper_const;
  std::shared_ptr<view_type> local_view = nullptr;

  std::size_t segment_size_ = 0;
  std::size_t row_size_ = 0;
  std::size_t max_row_size_ = 0;
  std::vector<std::size_t> row_offsets_;
  std::vector<std::size_t> row_sizes_;

  index_type *rows_data_ = nullptr;
  BackendT rows_backend_;

  distribution distribution_;
  dr::index<size_t> shape_;
  std::size_t nnz_;
  std::vector<segment_type> segments_;
  std::shared_ptr<distributed_vector<T>> vals_data_;
  std::shared_ptr<distributed_vector<I>> cols_data_;
};
} // namespace dr::mp
