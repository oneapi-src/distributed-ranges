// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <dr/detail/matrix_entry.hpp>
#include <dr/mp/containers/matrix_formats/csr_row_segment.hpp>
#include <dr/views/csr_matrix_view.hpp>
#include <fmt/core.h>

namespace dr::mp {
namespace __detail {
  template <typename T, typename V>
  class transform_fn_1 {
    public:
    using value_type = V;
    using index_type = T;
    transform_fn_1(std::size_t offset, std::size_t row_size, T* row_ptr):
      offset_(offset), row_size_(row_size), row_ptr_(row_ptr) {
      assert(offset_ == 0);
      }

    ~transform_fn_1() {
      destroyed = true;
    }
    template <typename P>
    auto operator()(P entry) const {
      assert(offset_ == 0);
      assert(!destroyed);
      auto [index, pair] = entry;
      auto [val, column] = pair;
      auto row = 0;
      // auto row = rng::distance(
      //           row_ptr_,
      //           std::upper_bound(row_ptr_, row_ptr_ + row_size_, offset_ + index) -
      //               1);
      dr::index<index_type> index_obj(row, column);
      value_type entry_obj(index_obj, val);
      return entry_obj;
    }
    private:
    bool destroyed = false;
    std::size_t offset_;
    std::size_t row_size_;
    T* row_ptr_;
  };
}

template <typename T, typename I, class BackendT = MpiBackend>
class csr_row_distribution {
public:
  using value_type = dr::matrix_entry<T, I>;
  using segment_type = csr_row_segment<csr_row_distribution>;
  using elem_type = T;
  using index_type = I;
  using difference_type = std::ptrdiff_t;

  csr_row_distribution(const csr_row_distribution &) = delete;
  csr_row_distribution &operator=(const csr_row_distribution &) = delete;
  csr_row_distribution(csr_row_distribution &&) { assert(false); }

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
  void fence() const {
    vals_backend_.fence();
    cols_backend_.fence();
  }
  template <typename C>
  auto local_gemv(C &res, T *vals, std::size_t vals_width) const {
    auto rank = cols_backend_.getrank();
    if (shape_[0] <= segment_size_ * rank)
      return;
    auto size = std::min(segment_size_, shape_[0] - segment_size_ * rank);
    auto vals_len = shape_[1];
    if (dr::mp::use_sycl()) {
      auto local_vals = vals_data_;
      auto local_cols = cols_data_;
      auto offset = val_offsets_[rank];
      auto real_segment_size = std::min(nnz_ - offset, val_sizes_[rank]);
      auto rows_data = dr::__detail::direct_iterator(
          dr::mp::local_segment(*rows_data_).begin());
      auto res_col_len = segment_size_;
      std::size_t wg = 32;
      while (vals_width * size * wg > INT_MAX) {
        // this check is necessary, because sycl does not permit ranges
        // exceeding integer limit
        wg /= 2;
      }
      assert(wg > 0);
      dr::mp::sycl_queue()
          .submit([&](auto &&h) {
            h.parallel_for(
                sycl::nd_range<1>(vals_width * size * wg, wg), [=](auto item) {
                  auto input_j = item.get_group(0) / size;
                  auto idx = item.get_group(0) % size;
                  auto local_id = item.get_local_id();
                  auto group_size = item.get_local_range(0);
                  std::size_t lower_bound = 0;
                  if (rows_data[idx] > offset) {
                    lower_bound = rows_data[idx] - offset;
                  }
                  std::size_t upper_bound = real_segment_size;
                  if (idx < size - 1) {
                    upper_bound = rows_data[idx + 1] - offset;
                  }
                  T sum = 0;
                  for (auto i = lower_bound + local_id; i < upper_bound;
                       i += group_size) {
                    auto colNum = local_cols[i];
                    auto matrixVal = vals[colNum + input_j * vals_len];
                    auto vectorVal = local_vals[i];
                    sum += matrixVal * vectorVal;
                  }

                  sycl::atomic_ref<T, sycl::memory_order::relaxed,
                                   sycl::memory_scope::device>
                      c_ref(res[idx + input_j * res_col_len]);
                  c_ref += sum;
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
        for (auto j = 0; j < vals_width; j++) {
          res[row_i + j * segment_size_] +=
              vals_data_[i] * vals[cols_data_[i] + j * vals_len];
        }
      }
    }
  }

  template <typename C>
  auto local_gemv_and_collect(std::size_t root, C &res, T *&vals,
                              std::size_t vals_width) const {
    assert(res.size() == shape_.first * vals_width);
    __detail::allocator<T> alloc;
    auto res_alloc = alloc.allocate(segment_size_ * vals_width);
    if (use_sycl()) {
      sycl_queue().fill(res_alloc, 0, segment_size_ * vals_width).wait();
    } else {
      std::fill(res_alloc, res_alloc + segment_size_ * vals_width, 0);
    }

    local_gemv(res_alloc, vals, vals_width);

    gather_gemv_vector(root, res, res_alloc, vals_width);
    fence();
    alloc.deallocate(res_alloc, segment_size_ * vals_width);
  }

private:
  friend csr_row_segment_iterator<csr_row_distribution>;

  template <typename C, typename A>
  void gather_gemv_vector(std::size_t root, C &res, A &partial_res,
                          std::size_t vals_width) const {
    auto communicator = default_comm();
    __detail::allocator<T> alloc;

    if (communicator.rank() == root) {
      auto scratch =
          alloc.allocate(segment_size_ * communicator.size() * vals_width);
      communicator.gather_typed(partial_res, scratch,
                                segment_size_ * vals_width, root);
      T *temp = nullptr;
      if (use_sycl()) {
        temp = new T[res.size()];
      }
      for (auto j = 0; j < communicator.size(); j++) {
        if (j * segment_size_ >= shape_.first) {
          break;
        }
        auto comm_segment_size =
            std::min(segment_size_, shape_.first - j * segment_size_);

        for (auto i = 0; i < vals_width; i++) {
          auto piece_start =
              scratch + j * vals_width * segment_size_ + i * segment_size_;

          if (use_sycl()) {
            __detail::sycl_copy(piece_start,
                                temp + shape_.first * i + j * segment_size_,
                                comm_segment_size);
          } else {
            std::copy(piece_start, piece_start + comm_segment_size,
                      res.begin() + shape_.first * i + j * segment_size_);
          }
        }
      }
      if (use_sycl()) {
        std::copy(temp, temp + res.size(), res.begin());
        delete[] temp;
      }
      alloc.deallocate(scratch,
                       segment_size_ * communicator.size() * vals_width);
    } else {
      communicator.gather_typed(partial_res, static_cast<T *>(nullptr),
                                segment_size_ * vals_width, root);
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
      segments_.emplace_back(
          this, segment_index++, val_sizes_[i],
          std::max(val_sizes_[i], static_cast<std::size_t>(1)));
    }
    fence();
    local_view = get_elem_view(vals_size_, cols_data_, vals_data_, rows_data_, rank);
  }

  static auto get_elem_view(std::size_t vals_size, 
  index_type *local_cols, 
  elem_type *local_vals,
   std::shared_ptr<distributed_vector<I>> rows_data, 
   std::size_t rank) {
    auto row_size = rows_data->segment_size();
    std::size_t offset = row_size * rank;
    auto local_vals_range = rng::subrange(local_vals, local_vals + vals_size);
    auto local_cols_range = rng::subrange(local_cols, local_cols + vals_size);
    auto local_rows = rows_data->segments()[rank].begin().local();
    auto zipped_results = rng::views::zip(local_vals_range, local_cols_range);
    auto enumerated_zipped = rng::views::enumerate(zipped_results);
    auto transformer = [=](auto entry){
      assert(offset == 0);
      auto [index, pair] = entry;
      auto [val, column] = pair;
      // auto row = 0;
      auto row = rng::distance(
                local_rows,
                std::upper_bound(local_rows, local_rows + row_size, offset + index) -
                    1);
      dr::index<index_type> index_obj(row, column);
      value_type entry_obj(index_obj, val);
      return entry_obj;
    };
    //__detail::transform_fn_1<index_type, value_type>(offset, row_size, local_rows);
    return rng::views::transform(enumerated_zipped, transformer);
  }

  using view_type = decltype(get_elem_view(0, nullptr, nullptr, std::shared_ptr<distributed_vector<I>>(nullptr),0));

  view_type local_view;
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
  std::vector<segment_type> segments_;
  std::shared_ptr<distributed_vector<I>> rows_data_;
};
} // namespace dr::mp
