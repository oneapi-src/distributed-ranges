// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <dr/mp/containers/matrix_formats/csr_matrix_segment.hpp>
#include<dr/sp/containers/matrix_entry.hpp>
#include<dr/sp/views/csr_matrix_view.hpp>

namespace dr::mp {

template <typename T, typename I, class BackendT = MpiBackend> 
class csr_matrix_distribution {
public:
    using value_type = dr::sp::matrix_entry<T, I>;
    using elem_type = T;
    using index_type = I;
    using difference_type = std::ptrdiff_t;

    csr_matrix_distribution(const csr_matrix_distribution &) = delete;
    csr_matrix_distribution &operator=(const csr_matrix_distribution &) = delete;
    csr_matrix_distribution(csr_matrix_distribution &&) { assert(false); }

    /// Constructor
    csr_matrix_distribution(dr::sp::csr_matrix_view<T, I> csr_view, distribution dist = distribution()) {
        init(csr_view, dist);
    }

    ~csr_matrix_distribution() {
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
      // if (dr::mp::use_sycl()) {

      // }
      // else {
        auto rank = rows_backend_.getrank();
        auto size = row_sizes_[rank];
        auto row_i = -1;
        auto position = segment_size_ * rank;
        auto current_row_position = rows_data_[0];
        auto local_vals = dr::mp::local_segment(*vals_data_);
        auto local_cols = dr::mp::local_segment(*cols_data_);

        for (int i = 0; i < segment_size_; i++) {
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
      // }
    }
    auto local_row_bounds(std::size_t rank) const {
      return std::pair<std::size_t, std::size_t>(row_offsets_[rank], row_offsets_[rank] + row_sizes_[rank]);
    }
private:
  friend csr_segment_iterator<csr_matrix_distribution>;
  std::size_t get_row_size(std::size_t rank) {
    return row_sizes_[rank];
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
      auto last_index = vals_data_->get_segment_offset(i + 1) - 1;
      auto lower_limit = std::distance(csr_view.rowptr_data(), std::upper_bound(csr_view.rowptr_data(), csr_view.rowptr_data() + shape_[0], first_index)) - 1;
      auto higher_limit = std::distance(csr_view.rowptr_data(), std::upper_bound(csr_view.rowptr_data(), csr_view.rowptr_data() + shape_[0], last_index));
      row_offsets_.push_back(lower_limit);
      row_sizes_.push_back(higher_limit - lower_limit);
    }

    auto lower_limit = row_offsets_[rank];
    row_size_ = row_sizes_[rank];
    if (row_size_ != get_row_size(rank)) {
      fmt::print("hmmmm? {} {} {} {}\n", rank, lower_limit, row_size_, get_row_size(rank));
    }
    
    rows_data_ = static_cast<I *>(rows_backend_.allocate(row_size_ * sizeof(I)));
    std::copy(csr_view.rowptr_data() + lower_limit, csr_view.rowptr_data() + lower_limit + row_size_, rows_data_);
    std::size_t segment_index = 0;
    segment_size_ = vals_data_->segment_size();
    assert(segment_size_ == cols_data_->segment_size());
    for (std::size_t i = 0; i < nnz_; i += segment_size_) {
      segments_.emplace_back(this, segment_index++,
                             std::min(segment_size_, nnz_ - i), segment_size_);
    }
    
    // for (int i = 0; i < row_size_; i++) {
    //   fmt::print("row, i, rank {} {} {}\n", rows_data_[i], i, rank);
    // }
    // fence();
    // for (int i = 0; i < vals_data_->segments()[rank].size(); i++) {
    //   fmt::print("val, col, i, rank {} {} {} {}\n", vals_data_->segments()[rank][i], cols_data_->segments()[rank][i],i, rank);
    // }

    fence();
  }


  std::size_t segment_size_ = 0;
  std::size_t row_size_ = 0;
  std::vector<std::size_t> row_offsets_;
  std::vector<std::size_t> row_sizes_;


  index_type *rows_data_ = nullptr;
  BackendT rows_backend_;

  distribution distribution_;
  dr::index<size_t> shape_;
  std::size_t nnz_;
  std::vector<csr_segment<csr_matrix_distribution>> segments_;
  std::shared_ptr<distributed_vector<T>> vals_data_;
  std::shared_ptr<distributed_vector<I>> cols_data_;
};
}