// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <dr/mp/containers/matrix_formats/csr_row_segment.hpp>
#include <dr/detail/matrix_entry.hpp>
#include <dr/views/csr_matrix_view.hpp>
#include <fmt/core.h>

namespace dr::mp {

template <typename T, typename I, class BackendT = MpiBackend> 
class csr_row_distribution {
public:
    using value_type = dr::matrix_entry<T, I>;
    using elem_type = T;
    using index_type = I;
    using difference_type = std::ptrdiff_t;

    csr_row_distribution(const csr_row_distribution &) = delete;
    csr_row_distribution &operator=(const csr_row_distribution &) = delete;
    csr_row_distribution(csr_row_distribution &&) { assert(false); }

    /// Constructor
    csr_row_distribution(dr::views::csr_matrix_view<T, I> csr_view, distribution dist = distribution()) {
        init(csr_view, dist);
    }

    ~csr_row_distribution() {
        if (!finalized()) {
        fence();
        if (vals_data_ != nullptr) {
            vals_backend_.deallocate(vals_data_, vals_size_ * sizeof(index_type));
            cols_backend_.deallocate(cols_data_, vals_size_ * sizeof(index_type));
        }

        //   delete halo_; TODO
        }
    }
    std::size_t get_id_in_segment(std::size_t offset) const {
      assert(offset < nnz_);
      auto pos_iter = std::upper_bound(val_offsets_.begin(), val_offsets_.end(), offset) - 1;
      return offset - *pos_iter;
    }
    std::size_t get_segment_from_offset(std::size_t offset) const {
      assert(offset < nnz_);
      auto pos_iter = std::upper_bound(val_offsets_.begin(), val_offsets_.end(), offset);
      return rng::distance(val_offsets_.begin(), pos_iter) - 1;
    }
    auto segments() const { return rng::views::all(segments_); }
    auto nnz() const {return nnz_;}
    auto shape() const {return shape_;}
    void fence() {
      vals_backend_.fence();
      cols_backend_.fence();
    }
template<typename C, typename A>
    auto local_gemv(C &res, A &vals) const {
      auto rank = cols_backend_.getrank();
      if (shape_[0] <= segment_size_ * rank) return;
      // if (dr::mp::use_sycl()) {

      // }
      // else {
        auto local_rows = dr::mp::local_segment(*rows_data_);
        auto size = std::min(segment_size_, shape_[0] - segment_size_ * rank);
        auto val_count = val_sizes_[rank];
        auto row_i = 0;
        auto position = val_offsets_[rank];
        auto current_row_position = local_rows[1];

        for (int i = 0; i < val_count; i++) {
          while (row_i + 1 < size && position + i >= current_row_position) {
            row_i++;
            current_row_position = local_rows[row_i + 1];
          }
          res[row_i] += vals_data_[i] * vals[cols_data_[i]];
        }
      // }
    }
 
    template<typename C, typename A>
    auto local_gemv_and_collect(std::size_t root, C &res, A &vals) const {
      assert(res.size() == shape_.first);
      __detail::allocator<T> alloc;
      auto res_alloc = alloc.allocate(segment_size_);
      local_gemv(res_alloc, vals);

      gather_gemv_vector(root, res, res_alloc);
      alloc.deallocate(res_alloc, segment_size_);
    }
private:
  friend csr_row_segment_iterator<csr_row_distribution>;
  
  template<typename C, typename A>
  void gather_gemv_vector(std::size_t root, C &res, A &partial_res) const {
      auto communicator = default_comm();
      __detail::allocator<T> alloc;
      if (communicator.rank() == root) {
          auto scratch = alloc.allocate(segment_size_ * default_comm().size());
          communicator.gather(partial_res, scratch, segment_size_, root);
          std::copy(scratch, scratch + shape_.first, res.begin());
          alloc.deallocate(scratch, segment_size_ * communicator.size());
      }
      else {
          communicator.gather(partial_res, static_cast<T*>(nullptr), segment_size_, root);
      }
  }
  void init(dr::views::csr_matrix_view<T, I> csr_view, auto dist) {
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


    auto rank = vals_backend_.getrank();
    rows_data_ = std::make_shared<distributed_vector<I>>(shape_.first);

    dr::mp::copy(std::ranges::subrange(csr_view.rowptr_data(), csr_view.rowptr_data() + shape_.first), rows_data_->begin());
 
    assert(*csr_view.rowptr_data() == 0);
    for (int i = 0; i < default_comm().size(); i++) {
      auto first_index = rows_data_->get_segment_offset(i);
      if (first_index > shape_.first) {
        val_offsets_.push_back(nnz_);
        val_sizes_.push_back(0);
        continue;
      }
      std::size_t lower_limit = csr_view.rowptr_data()[first_index];
      std::size_t higher_limit = nnz_;
      if (rows_data_->get_segment_offset(i + 1) < shape_.first) {
        auto last_index = rows_data_->get_segment_offset(i + 1);
        higher_limit = csr_view.rowptr_data()[last_index];
      }
      val_offsets_.push_back(lower_limit);
      val_sizes_.push_back(higher_limit - lower_limit);
    }

    auto lower_limit = val_offsets_[rank];
    vals_size_ = std::max(val_sizes_[rank], static_cast<std::size_t>(1));
    // fmt::print("dfsa {} {} {} {}\n", vals_size_, val_sizes_[rank],lower_limit, rank);
    
    cols_data_ = static_cast<I *>(cols_backend_.allocate(vals_size_ * sizeof(I)));
    vals_data_ = static_cast<T *>(vals_backend_.allocate(vals_size_ * sizeof(T)));
    std::copy(csr_view.values_data() + lower_limit, csr_view.values_data() + lower_limit + vals_size_, vals_data_);
    std::copy(csr_view.colind_data() + lower_limit, csr_view.colind_data() + lower_limit + vals_size_, cols_data_);

    std::size_t segment_index = 0;
    segment_size_ = rows_data_->segment_size();
    for (std::size_t i = 0; i < default_comm().size(); i++) {
      //TODO fix segment creation, to include proper sizes, basing on val_offsets;
      segments_.emplace_back(this, segment_index++, val_sizes_[i], std::max(val_sizes_[i], static_cast<std::size_t>(1)));
    }
    // if (rank == 0) {
    //   int ax = 0;
    //   for (auto x: val_offsets_) {
    //     fmt::print("{} {}\n", ax++, x);
    //   }
    //   for (int i = 0; i < 49; i++) {
    //     fmt::print("{} {}\n", i, get_segment_from_offset(i));
    //   }
    // }
    // fmt::print(" {} {} {} {}\n",get_segment_from_offset(47), get_segment_from_offset(48), get_segment_from_offset(49), get_segment_from_offset(50));
    // for (int i = 0; i < vals_size_; i++) {
    //   fmt::print("col, val, i, rank {} {} {} {}\n", cols_data_[i], vals_data_[i], i, rank);
    // }
    // fence();
    // if (rank < rows_data_->segments().size()) {
    //   for (int i = 0; i < rows_data_->segments()[rank].size(); i++) {
    //     fmt::print("row, i, rank {} {} {}\n", rows_data_->segments()[rank][i], i, rank);
    //   }
    // }
    fence();
  }


  std::size_t segment_size_ = 0;
  std::size_t vals_size_ = 0;
  std::vector<std::size_t> val_offsets_;
  std::vector<std::size_t> val_sizes_;


  index_type *cols_data_ = nullptr;
  BackendT cols_backend_;
  
  elem_type *vals_data_ = nullptr;
  BackendT vals_backend_;

  distribution distribution_;
  dr::index<size_t> shape_;
  std::size_t nnz_;
  std::vector<csr_row_segment<csr_row_distribution>> segments_;
  std::shared_ptr<distributed_vector<I>> rows_data_;
};
}