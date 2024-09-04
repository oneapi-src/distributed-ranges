// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <dr/mp/containers/matrix_formats/csr_eq_segment.hpp>
#include<dr/detail/matrix_entry.hpp>
#include<dr/views/csr_matrix_view.hpp>

namespace dr::mp {

template <typename T, typename I, class BackendT = MpiBackend> 
class csr_eq_distribution {
public:
    using value_type = dr::matrix_entry<T, I>;
    using elem_type = T;
    using index_type = I;
    using difference_type = std::ptrdiff_t;

    csr_eq_distribution(const csr_eq_distribution &) = delete;
    csr_eq_distribution &operator=(const csr_eq_distribution &) = delete;
    csr_eq_distribution(csr_eq_distribution &&) { assert(false); }

    /// Constructor
    csr_eq_distribution(dr::views::csr_matrix_view<T, I> csr_view, distribution dist = distribution(), std::size_t root = 0) {
        init(csr_view, dist, root);
    }

    ~csr_eq_distribution() {
        if (!finalized()) {
        fence();
        if (rows_data_ != nullptr) {
            rows_backend_.deallocate(rows_data_, row_size_ * sizeof(index_type));
        }

        //   delete halo_; TODO
        }
    }
    std::size_t get_id_in_segment(std::size_t offset) const {
      return offset % segment_size_;
    }
    std::size_t get_segment_from_offset(std::size_t offset) const {
      return offset / segment_size_;
    }
    auto segments() const { return rng::views::all(segments_); }
    auto nnz() const {return nnz_;}
    auto shape() const {return shape_;}
    void fence() {
      rows_backend_.fence();
    }

    template<typename C, typename A>
    auto local_gemv(C &res, A &vals) const {
      auto rank = rows_backend_.getrank();
      if (nnz_ <= segment_size_ * rank) {
        return;
      }
      auto size = row_sizes_[rank];
      if (dr::mp::use_sycl()) {
        auto localVals = dr::__detail::direct_iterator(dr::mp::local_segment(*vals_data_).begin());
        auto localCols = dr::__detail::direct_iterator(dr::mp::local_segment(*cols_data_).begin());
        auto offset = rank * segment_size_;
        auto real_segment_size = std::min(nnz_ - rank * segment_size_, segment_size_);
        auto local_data = rows_data_;
        // dr::mp::sycl_queue().submit([&](auto& cgh) {
        //   cgh.parallel_for(sycl::range<1> { real_segment_size },
        //                   [=](auto idx) {
        //                     auto colNum = localCols[idx];
        //                     auto matrixVal = vals[colNum];
        //                     auto vectorVal = localVals[idx];
        //                     auto row = std::distance(std::upper_bound(local_data, local_data + row_size, offset + idx), local_data) - 1;
        //                     *(res + row) += matrixVal * vectorVal;
        //                   });
        // }).wait();
        auto one_computation_size = (real_segment_size + max_row_size_ - 1) / max_row_size_;
        auto row_size = row_size_;
        dr::mp::sycl_queue().submit([&](auto& cgh) {
          cgh.parallel_for(sycl::range<1> { max_row_size_ },
                          [=](auto idx) {
                            std::size_t lower_bound = one_computation_size * idx;
                            std::size_t upper_bound = std::min(one_computation_size * (idx + 1), real_segment_size);
                            std::size_t position = lower_bound + offset;
                            std::size_t first_row = std::distance(local_data, std::upper_bound(local_data, local_data + row_size, position) - 1);
                            auto row = first_row;
                            T sum = 0;
                            for (auto i = lower_bound; i < upper_bound; i++) {
                              while (row + 1 < row_size && local_data[row + 1] <= offset + i) {
                                sycl::atomic_ref<T, sycl::memory_order::relaxed,
                                                sycl::memory_scope::device>
                                    c_ref(res[row]);
                                c_ref += sum;
                                row++;
                                sum = 0;
                              }
                              auto colNum = localCols[i];
                              auto matrixVal = vals[colNum];
                              auto vectorVal = localVals[i];
                              
                              sum += matrixVal * vectorVal;
                            }
                            sycl::atomic_ref<T, sycl::memory_order::relaxed,
                                            sycl::memory_scope::device>
                                c_ref(res[row]);
                            c_ref += sum;
                          });
        }).wait();
      }
      else {
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
          res[row_i] += local_vals[i] * vals[local_cols[i]];
        }

        // fmt::print("offset, rank {} {}\n", row_offsets_[ rows_backend_.getrank()],  rows_backend_.getrank());
        // for (int i = 0; i < size; i++) {
        //   fmt::print("ledata, rank, i {} {} {}\n", res[i], rows_backend_.getrank(), i);
        // }
      }
    }
 
    template<typename C, typename A>
    auto local_gemv_and_collect(std::size_t root, C &res, A &vals) const {
      assert(res.size() == shape_.first);
      __detail::allocator<T> alloc;
      auto res_alloc = alloc.allocate(max_row_size_);
      for (auto i = 0; i < max_row_size_; i++) {
        res_alloc[i] = 0;
      }
      auto begin = std::chrono::high_resolution_clock::now();
      local_gemv(res_alloc, vals);
      auto end = std::chrono::high_resolution_clock::now();
      double duration = std::chrono::duration<double>(end - begin).count();
      fmt::print("eq gemv time {}\n", duration * 1000);

      gather_gemv_vector(root, res, res_alloc);
      alloc.deallocate(res_alloc, max_row_size_);
    }
private:
  friend csr_eq_segment_iterator<csr_eq_distribution>;
  
  template<typename C, typename A>
  void gather_gemv_vector(std::size_t root, C &res, A &partial_res) const {
      auto communicator = default_comm();
      __detail::allocator<T> alloc;
      if (communicator.rank() == root) {
          auto gathered_res = alloc.allocate(max_row_size_ * communicator.size());
          communicator.gather(partial_res, gathered_res, max_row_size_, root);
          rng::fill(res, 0);
          
        // auto begin = std::chrono::high_resolution_clock::now();
          for (auto i = 0; i < communicator.size(); i++) {
              auto first_row = row_offsets_[i];
              auto last_row = row_offsets_[i] + row_sizes_[i];
              for (auto j = first_row; j < last_row; j++) {
                  res[j] += gathered_res[max_row_size_ * i + j - first_row];
              }
          }
        // auto end = std::chrono::high_resolution_clock::now();
        // double duration = std::chrono::duration<double>(end - begin).count();
        // fmt::print("gather time {}\n", duration);
          alloc.deallocate(gathered_res, max_row_size_ * communicator.size());
      }
      else {
          communicator.gather(partial_res, static_cast<T*>(nullptr), max_row_size_, root);
      }
  }

  std::size_t get_row_size(std::size_t rank) {
    return row_sizes_[rank];
  }

  void init(dr::views::csr_matrix_view<T, I> csr_view, auto dist, std::size_t root) {
    distribution_ = dist;
    auto rank = rows_backend_.getrank();

    std::size_t initial_data[3];
    if (root == rank) {
      initial_data[0] = csr_view.size();
      initial_data[1] = csr_view.shape().first;
      initial_data[2] = csr_view.shape().second;
      default_comm().bcast(initial_data, sizeof(std::size_t) * 3, root);
    }
    else {
      default_comm().bcast(initial_data, sizeof(std::size_t) * 3, root);
    }

    nnz_ = initial_data[0];
    shape_ = {initial_data[1], initial_data[2]};
    vals_data_ = std::make_shared<distributed_vector<T>>(nnz_);
    cols_data_ = std::make_shared<distributed_vector<I>>(nnz_);
    dr::mp::copy(root, std::ranges::subrange(csr_view.values_data(), csr_view.values_data() + nnz_), vals_data_->begin());
    dr::mp::copy(root, std::ranges::subrange(csr_view.colind_data(), csr_view.colind_data() + nnz_), cols_data_->begin());

    auto row_info_size = default_comm().size() * 2 + 1;
    __detail::allocator<size_t> alloc;
    std::size_t* row_information = new std::size_t[row_info_size];
    row_offsets_.reserve(default_comm().size());
    row_sizes_.reserve(default_comm().size());
    if (root == default_comm().rank()) {
      for (int i = 0; i < default_comm().size(); i++) {
        auto first_index = vals_data_->get_segment_offset(i);
        auto last_index = vals_data_->get_segment_offset(i + 1) - 1;
        auto lower_limit = std::distance(csr_view.rowptr_data(), std::upper_bound(csr_view.rowptr_data(), csr_view.rowptr_data() + shape_[0], first_index)) - 1;
        auto higher_limit = std::distance(csr_view.rowptr_data(), std::upper_bound(csr_view.rowptr_data(), csr_view.rowptr_data() + shape_[0], last_index));
        row_offsets_.push_back(lower_limit);
        row_sizes_.push_back(higher_limit - lower_limit);
        row_information[i] = lower_limit;
        row_information[default_comm().size() + i] = higher_limit - lower_limit;
        max_row_size_ = std::max(max_row_size_, row_sizes_.back());
      }
      row_information[default_comm().size() * 2] = max_row_size_;
      default_comm().bcast(row_information, sizeof(std::size_t) * row_info_size, root);
    }
    else {
      default_comm().bcast(row_information, sizeof(std::size_t) * row_info_size, root);
      for (int i = 0; i < default_comm().size(); i++) {
        row_offsets_.push_back(row_information[i]);
        row_sizes_.push_back(row_information[default_comm().size() + i]);
      }
      max_row_size_ = row_information[default_comm().size() * 2];
    }
    delete[] row_information;
    row_size_ = std::max(row_sizes_[rank], static_cast<std::size_t>(1));
    rows_data_ = static_cast<I *>(rows_backend_.allocate(row_size_ * sizeof(I)));

    fence();
    if (rank == root) {
      for (std::size_t i = 0; i < default_comm().size(); i++) {
        auto lower_limit = row_offsets_[i];
        auto row_size = row_sizes_[i];
        if (row_size > 0) {
        rows_backend_.putmem(csr_view.rowptr_data() + lower_limit, 0, row_size * sizeof(I), i);
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
    fence();
    // for (int i = 0; i < row_size_; i++) {
    //   fmt::print("row, i, rank {} {} {}\n", rows_data_[i], i, rank);
    // }
    // fence();
    // for (int i = 0; i < vals_data_->segments()[rank].size(); i++) {
    //   fmt::print("val, col, i, rank {} {} {} {}\n", vals_data_->segments()[rank][i], cols_data_->segments()[rank][i],i, rank);
    // }

  }


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
  std::vector<csr_eq_segment<csr_eq_distribution>> segments_;
  std::shared_ptr<distributed_vector<T>> vals_data_;
  std::shared_ptr<distributed_vector<I>> cols_data_;
};
}