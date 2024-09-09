// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <dr/detail/matrix_entry.hpp>
#include <dr/mp/containers/matrix_formats/csr_row_segment.hpp>
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
  csr_row_distribution(dr::views::csr_matrix_view<T, I> csr_view,
                       distribution dist = distribution(),
                       std::size_t root = 0) {
    init(csr_view, dist, root);
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
    auto pos_iter =
        std::upper_bound(val_offsets_.begin(), val_offsets_.end(), offset) - 1;
    return offset - *pos_iter;
  }
  std::size_t get_segment_from_offset(std::size_t offset) const {
    assert(offset < nnz_);
    auto pos_iter =
        std::upper_bound(val_offsets_.begin(), val_offsets_.end(), offset);
    return rng::distance(val_offsets_.begin(), pos_iter) - 1;
  }
  auto segments() const { return rng::views::all(segments_); }
  auto nnz() const { return nnz_; }
  auto shape() const { return shape_; }
  void fence() {
    vals_backend_.fence();
    cols_backend_.fence();
  }
  template <typename C, typename A> auto local_gemv(C &res, A &vals) const {
    auto rank = cols_backend_.getrank();
    if (shape_[0] <= segment_size_ * rank)
      return;
    auto size = std::min(segment_size_, shape_[0] - segment_size_ * rank);
    if (dr::mp::use_sycl()) {
      auto local_vals = vals_data_;
      auto local_cols = cols_data_;
      auto offset = val_offsets_[rank];
      auto real_segment_size = std::min(nnz_ - offset, val_sizes_[rank]);
      auto rows_data = dr::__detail::direct_iterator(
          dr::mp::local_segment(*rows_data_).begin());
      dr::mp::sycl_queue()
          .submit([&](auto &cgh) {
            cgh.parallel_for(sycl::range<1>{size}, [=](auto idx) {
              std::size_t lower_bound = 0;
              T sum = 0;
              if (rows_data[idx] > offset) {
                lower_bound = rows_data[idx] - offset;
              }
              std::size_t upper_bound = real_segment_size;
              if (idx < size - 1) {
                upper_bound = rows_data[idx + 1] - offset;
              }
              for (auto i = lower_bound; i < upper_bound; i++) {
                auto colNum = local_cols[i];
                auto matrixVal = vals[colNum];
                auto vectorVal = local_vals[i];
                sum += matrixVal * vectorVal;
              }
              *(res + idx) += sum;
            });
          })
          .wait();
    } else {
      auto local_rows = dr::mp::local_segment(*rows_data_);
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
    }
  }

  template <typename C, typename A>
  auto local_gemv_and_collect(std::size_t root, C &res, A &vals) const {
    assert(res.size() == shape_.first);
    __detail::allocator<T> alloc;
    auto res_alloc = alloc.allocate(segment_size_);
    for (auto i = 0; i < segment_size_; i++) {
      res_alloc[i] = 0;
    }

    // auto begin = std::chrono::high_resolution_clock::now();
    local_gemv(res_alloc, vals);
    // auto end = std::chrono::high_resolution_clock::now();
    // double duration = std::chrono::duration<double>(end - begin).count();
    // fmt::print("rows gemv time {}\n", duration * 1000);

    gather_gemv_vector(root, res, res_alloc);
    alloc.deallocate(res_alloc, segment_size_);
  }

private:
  friend csr_row_segment_iterator<csr_row_distribution>;

  template <typename C, typename A>
  void gather_gemv_vector(std::size_t root, C &res, A &partial_res) const {
    auto communicator = default_comm();
    __detail::allocator<T> alloc;
    if (communicator.rank() == root) {
      auto scratch = alloc.allocate(segment_size_ * default_comm().size());
      communicator.gather(partial_res, scratch, segment_size_, root);
      std::copy(scratch, scratch + shape_.first, res.begin());
      alloc.deallocate(scratch, segment_size_ * communicator.size());
    } else {
      communicator.gather(partial_res, static_cast<T *>(nullptr), segment_size_,
                          root);
    }
  }
  void init(dr::views::csr_matrix_view<T, I> csr_view, auto dist,
            std::size_t root) {
    distribution_ = dist;
    auto rank = vals_backend_.getrank();

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

    rows_data_ = std::make_shared<distributed_vector<I>>(shape_.first);

    dr::mp::copy(root,
                 std::ranges::subrange(csr_view.rowptr_data(),
                                       csr_view.rowptr_data() + shape_.first),
                 rows_data_->begin());

    auto row_info_size = default_comm().size() * 2;
    std::size_t *val_information = new std::size_t[row_info_size];
    val_offsets_.reserve(row_info_size);
    val_sizes_.reserve(row_info_size);
    if (rank == root) {
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
        val_information[i] = lower_limit;
        val_information[i + default_comm().size()] = higher_limit - lower_limit;
      }
      default_comm().bcast(val_information, sizeof(std::size_t) * row_info_size,
                           root);
    } else {
      default_comm().bcast(val_information, sizeof(std::size_t) * row_info_size,
                           root);
      for (int i = 0; i < default_comm().size(); i++) {
        val_offsets_.push_back(val_information[i]);
        val_sizes_.push_back(val_information[default_comm().size() + i]);
      }
    }
    delete[] val_information;
    vals_size_ = std::max(val_sizes_[rank], static_cast<std::size_t>(1));
    // fmt::print("dfsa {} {} {} {}\n", vals_size_,
    // val_sizes_[rank],lower_limit, rank);

    cols_data_ =
        static_cast<I *>(cols_backend_.allocate(vals_size_ * sizeof(I)));
    vals_data_ =
        static_cast<T *>(vals_backend_.allocate(vals_size_ * sizeof(T)));

    fence();
    if (rank == root) {
      for (std::size_t i = 0; i < default_comm().size(); i++) {
        auto lower_limit = val_offsets_[i];
        auto row_size = val_sizes_[i];
        if (row_size > 0) {
          vals_backend_.putmem(csr_view.values_data() + lower_limit, 0,
                               row_size * sizeof(T), i);
          cols_backend_.putmem(csr_view.colind_data() + lower_limit, 0,
                               row_size * sizeof(I), i);
        }
      }
    }

    std::size_t segment_index = 0;
    segment_size_ = rows_data_->segment_size();
    for (std::size_t i = 0; i < default_comm().size(); i++) {
      // TODO fix segment creation, to include proper sizes, basing on
      // val_offsets;
      segments_.emplace_back(
          this, segment_index++, val_sizes_[i],
          std::max(val_sizes_[i], static_cast<std::size_t>(1)));
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
    // fmt::print(" {} {} {} {}\n",get_segment_from_offset(47),
    // get_segment_from_offset(48), get_segment_from_offset(49),
    // get_segment_from_offset(50)); for (int i = 0; i < vals_size_; i++) {
    //   fmt::print("col, val, i, rank {} {} {} {}\n", cols_data_[i],
    //   vals_data_[i], i, rank);
    // }
    // fence();
    // if (rank < rows_data_->segments().size()) {
    //   for (int i = 0; i < rows_data_->segments()[rank].size(); i++) {
    //     fmt::print("row, i, rank {} {} {}\n",
    //     rows_data_->segments()[rank][i], i, rank);
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
} // namespace dr::mp
