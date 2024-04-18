// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#ifdef SYCL_LANGUAGE_VERSION
#include <dr/shp/init.hpp>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>
#endif

#include <mpi.h>

#include <algorithm>
#include <utility>

#include <dr/concepts/concepts.hpp>
#include <dr/detail/logger.hpp>
#include <dr/detail/onedpl_direct_iterator.hpp>
#include <dr/detail/ranges_shim.hpp>
#include <dr/mhp/global.hpp>

namespace dr::mhp {

namespace __detail {

template <typename T> class buffer {
public:
  using value_type = T;
  std::size_t size() { return size_; }
  T *data() { return data_; }
  T *begin() { return data_; }
  T *end() { return data_ + size_; }

  T *resize(std::size_t cnt) {
    if (cnt == size_)
      return data_;

    if (mhp::use_sycl()) {
#ifdef SYCL_LANGUAGE_VERSION
      if (cnt == 0) {
        sycl::free(data_, sycl_queue());
        data_ = nullptr;
      } else {
        T *newdata = sycl::malloc<T>(cnt, sycl_queue(), sycl_mem_kind());
        assert(newdata != nullptr);
        sycl_queue()
            .copy<T>(data_, newdata, (size_ < cnt) ? size_ : cnt)
            .wait();
        sycl::free(data_, sycl_queue());
        data_ = newdata;
      }
#else
      assert(false);
#endif
    } else {
      if (cnt == 0) {
        free(data_);
        data_ = nullptr;
      } else {
        T *newdata = static_cast<T *>(malloc(cnt * sizeof(T)));
        memcpy(newdata, data_, size_ * sizeof(T));
        free(data_);
        data_ = newdata;
      }
    }
    size_ = cnt;
    return data_;
  }

  void replace(buffer &other) {
    if (mhp::use_sycl()) {
#ifdef SYCL_LANGUAGE_VERSION
      if (data_ != nullptr)
        sycl::free(data_, sycl_queue());
#else
      assert(false);
#endif
    } else {
      if (data_ != nullptr)
        free(data_);
    }
    data_ = rng::data(other);
    size_ = rng::size(other);
    other.data_ = nullptr;
    other.size_ = 0;
  }

  ~buffer() {
    if (data_ != nullptr) {
      if (mhp::use_sycl()) {
#ifdef SYCL_LANGUAGE_VERSION
        sycl::free(this->data_, sycl_queue());
#else
        assert(false);
#endif
      } else {
        free(data_);
      }
    }
    data_ = nullptr;
    size_ = 0;
  }

  buffer(std::size_t cnt) : size_(cnt) {
    if (cnt > 0) {
      if (mhp::use_sycl()) {
#ifdef SYCL_LANGUAGE_VERSION
        data_ = sycl::malloc<T>(cnt, sycl_queue(), sycl_mem_kind());
#else
        assert(false);
#endif
      } else {
        data_ = static_cast<T *>(malloc(cnt * sizeof(T)));
      }
      assert(data_ != nullptr);
    }
  }

private:
  T *data_ = nullptr;
  std::size_t size_ = 0;
}; // class buffer

template <typename R, typename Compare> void local_sort(R &r, Compare &&comp) {
  if (rng::size(r) >= 2) {
    if (mhp::use_sycl()) {
#ifdef SYCL_LANGUAGE_VERSION
      auto policy = dpl_policy();
      auto &&local_segment = dr::ranges::__detail::local(r);
      DRLOG("GPU dpl::sort(), size {}", rng::size(r));
      oneapi::dpl::sort(
          policy, dr::__detail::direct_iterator(rng::begin(local_segment)),
          dr::__detail::direct_iterator(rng::end(local_segment)), comp);
#else
      assert(false);
#endif
    } else {
      DRLOG("cpu rng::sort, size {}", rng::size(r));
      rng::sort(rng::begin(r), rng::end(r), comp);
    }
  }
}

/* elements of dist_sort */
template <typename valT, typename Compare, typename Seg>
void splitters(Seg &lsegment, Compare &&comp, auto &vec_split_i,
               auto &vec_split_s) {
  const std::size_t _comm_size = default_comm().size(); // dr-style ignore

  assert(rng::size(vec_split_i) == _comm_size);
  assert(rng::size(vec_split_s) == _comm_size);

  std::vector<valT> vec_lmedians(_comm_size + 1);
  std::vector<valT> vec_gmedians((_comm_size + 1) * _comm_size);

  const double _step_m = static_cast<double>(rng::size(lsegment)) /
                         static_cast<double>(_comm_size);

  /* calculate splitting values and indices - find n-1 dividers splitting
   * each segment into equal parts */
  if (mhp::use_sycl()) {
#ifdef SYCL_LANGUAGE_VERSION
    std::vector<sycl::event> events;

    for (std::size_t _i = 0; _i < rng::size(vec_lmedians) - 1; _i++) {
      assert(_i * _step_m < rng::size(lsegment));
      sycl::event ev = sycl_queue().memcpy(
          &vec_lmedians[_i], &lsegment[_i * _step_m], sizeof(valT));
      events.emplace_back(ev);
    }
    sycl::event ev =
        sycl_queue().memcpy(&vec_lmedians[rng::size(vec_lmedians) - 1],
                            &lsegment[rng::size(lsegment) - 1], sizeof(valT));
    events.emplace_back(ev);
    sycl::event::wait(events);
#else
    assert(false);
#endif
  } else {
    for (std::size_t _i = 0; _i < rng::size(vec_lmedians) - 1; _i++) {
      assert(_i * _step_m < rng::size(lsegment));
      vec_lmedians[_i] = lsegment[_i * _step_m];
    }
    vec_lmedians.back() = lsegment.back();
  }

  default_comm().all_gather(vec_lmedians, vec_gmedians);
  rng::sort(rng::begin(vec_gmedians), rng::end(vec_gmedians), comp);

  std::vector<valT> vec_split_v(_comm_size - 1);

  for (std::size_t _i = 0; _i < _comm_size - 1; _i++) {
    auto global_median_idx = (_i + 1) * (_comm_size + 1) - 1;
    assert(global_median_idx < rng::size(vec_gmedians));
    vec_split_v[_i] = vec_gmedians[global_median_idx];
  }

  if (mhp::use_sycl()) {
#ifdef SYCL_LANGUAGE_VERSION
    auto &&local_policy = dpl_policy();
    sycl::queue q = sycl_queue();

    auto lsb = dr::__detail::direct_iterator(rng::begin(lsegment));
    auto lse = dr::__detail::direct_iterator(rng::end(lsegment));

    oneapi::dpl::lower_bound(local_policy, lsb, lse, vec_split_v.begin(),
                             vec_split_v.end(), vec_split_i.begin() + 1, comp);

    for (std::size_t _i = 1; _i < vec_split_i.size(); _i++) {
      vec_split_s[_i - 1] = vec_split_i[_i] - vec_split_i[_i - 1];
    }
    vec_split_s.back() = rng::size(lsegment) - vec_split_i.back();

#else
    assert(false);
#endif
  } else {
    auto first = lsegment.begin();
    for (std::size_t i = 1; i <= rng::size(vec_split_v); i++) {
      auto idx = vec_split_v[i - 1];
      auto lower =
          std::lower_bound(lsegment.begin(), lsegment.end(), idx, comp);
      auto idx_lower = rng::distance(lsegment.begin(), lower);
      auto chunk_size = rng::distance(first, lower);
      vec_split_i[i] = idx_lower;
      vec_split_s[i - 1] = chunk_size;
      first = lower;
    }
    vec_split_s.back() = rng::size(lsegment) - vec_split_i.back();
  }
}

template <typename valT>
void shift_data(const int64_t shift_left, const int64_t shift_right,
                buffer<valT> &vec_recvdata, buffer<valT> &vec_left,
                buffer<valT> &vec_right) {
  const std::size_t _comm_rank = default_comm().rank();

  MPI_Request req_l, req_r;
  MPI_Status stat_l, stat_r;

  assert(static_cast<int64_t>(rng::size(vec_left)) == std::max(0L, shift_left));
  assert(static_cast<int64_t>(rng::size(vec_right)) ==
         std::max(0L, shift_right));

  if (static_cast<int64_t>(rng::size(vec_recvdata)) < -shift_left) {
    // Too little data in recv buffer to shift left - first get from right,
    // then send left
    DRLOG("Get from right first, recvdata size {} shift left {}",
          rng::size(vec_recvdata), shift_left);

    assert(shift_right > 0);

    default_comm().irecv(rng::data(vec_right), rng::size(vec_right),
                         _comm_rank + 1, &req_r);
    MPI_Wait(&req_r, &stat_r);

    std::size_t old_size = rng::size(vec_recvdata);
    vec_recvdata.resize(rng::size(vec_recvdata) + shift_right);

    assert(rng::size(vec_right) <= rng::size(vec_recvdata) - old_size);
    if (mhp::use_sycl()) {
#ifdef SYCL_LANGUAGE_VERSION
      sycl_queue()
          .copy<valT>(rng::data(vec_right), rng::data(vec_recvdata) + old_size,
                      rng::size(vec_right))
          .wait();
#else
      assert(false);
#endif
    } else {
      std::copy(rng::begin(vec_right), rng::end(vec_right),
                rng::begin(vec_recvdata) + old_size);
    }
    vec_right.resize(0);

    default_comm().isend(rng::data(vec_recvdata), -shift_left, _comm_rank - 1,
                         &req_l);
    MPI_Wait(&req_l, &stat_l);

  } else if (static_cast<int64_t>(rng::size(vec_recvdata)) < -shift_right) {
    // Too little data in buffer to shift right - first get from left, then
    // send right
    // ** This will never happen, because values eq to split go right
    DRLOG(
        "Too little data in buffer to shift right - this should never happen");
    assert(false);

  } else {
    // enough data in recv buffer
    if (shift_left < 0) {
      default_comm().isend(rng::data(vec_recvdata), -shift_left, _comm_rank - 1,
                           &req_l);
    } else if (shift_left > 0) {
      assert(shift_left == static_cast<int64_t>(rng::size(vec_left)));
      default_comm().irecv(rng::data(vec_left), rng::size(vec_left),
                           _comm_rank - 1, &req_l);
    }
    if (shift_right > 0) {
      assert(shift_right == static_cast<int64_t>(rng::size(vec_right)));
      default_comm().irecv(rng::data(vec_right), rng::size(vec_right),
                           _comm_rank + 1, &req_r);
    } else if (shift_right < 0) {
      default_comm().isend(rng::data(vec_recvdata) + rng::size(vec_recvdata) +
                               shift_right,
                           -shift_right, _comm_rank + 1, &req_r);
    }
    if (shift_left != 0)
      MPI_Wait(&req_l, &stat_l);
    if (shift_right != 0)
      MPI_Wait(&req_r, &stat_r);
  }
}

template <typename valT>
void copy_results(auto &lsegment, const int64_t shift_left,
                  const int64_t shift_right, buffer<valT> &vec_recvdata,
                  buffer<valT> &vec_left, buffer<valT> &vec_right) {
  const std::size_t invalidate_left = std::max(-shift_left, 0L);
  const std::size_t invalidate_right = std::max(-shift_right, 0L);

  const std::size_t size_l = rng::size(vec_left);
  const std::size_t size_r = rng::size(vec_right);
  const std::size_t size_d =
      rng::size(vec_recvdata) - (invalidate_left + invalidate_right);

  if (mhp::use_sycl()) {
#ifdef SYCL_LANGUAGE_VERSION
    sycl::event e_l, e_d, e_r;

    if (size_l > 0) {
      assert(size_l <= rng::size(lsegment));
      e_l = sycl_queue().copy(rng::data(vec_left), rng::data(lsegment), size_l);
    }
    if (size_r > 0) {
      assert(size_l + size_d + size_r <= rng::size(lsegment));
      e_r = sycl_queue().copy(rng::data(vec_right),
                              rng::data(lsegment) + size_l + size_d, size_r);
    }
    if (size_d > 0) {
      assert(size_l + size_d <= rng::size(lsegment));
      assert(invalidate_left + size_d <= rng::size(vec_recvdata));
      e_d = sycl_queue().copy(rng::data(vec_recvdata) + invalidate_left,
                              rng::data(lsegment) + size_l, size_d);
    }
    if (size_l > 0)
      e_l.wait();
    if (size_r > 0)
      e_r.wait();
    if (size_d > 0)
      e_d.wait();

#else
    assert(false);
#endif
  } else {
    if (size_l > 0) {
      assert(size_l <= rng::size(lsegment));
      std::copy(rng::begin(vec_left), rng::end(vec_left), rng::begin(lsegment));
    }
    if (size_r > 0) {
      assert(size_l + size_d + size_r <= rng::size(lsegment));
      std::copy(rng::begin(vec_right), rng::end(vec_right),
                rng::begin(lsegment) + size_l + size_d);
    }
    if (size_d > 0) {
      assert(size_l + size_d <= rng::size(lsegment));
      assert(invalidate_left + size_d <= rng::size(vec_recvdata));
      std::copy(rng::begin(vec_recvdata) + invalidate_left,
                rng::begin(vec_recvdata) + invalidate_left + size_d,
                rng::begin(lsegment) + size_l);
    }
  }
}

template <dr::distributed_range R, typename Compare>
void dist_sort(R &r, Compare &&comp) {
  using valT = typename R::value_type;

  const std::size_t _comm_rank = default_comm().rank();
  const std::size_t _comm_size = default_comm().size(); // dr-style ignore

  auto &&lsegment = local_segment(r);

  std::vector<std::size_t> vec_split_i(_comm_size, 0);
  std::vector<std::size_t> vec_split_s(_comm_size, 0);
  std::vector<std::size_t> vec_rsizes(_comm_size, 0);
  std::vector<std::size_t> vec_rindices(_comm_size, 0);
  std::vector<std::size_t> vec_recv_elems(_comm_size, 0);
  std::size_t _total_elems = 0;

  DRLOG("Rank {}: Dist sort, local segment size {}", default_comm().rank(),
        rng::size(lsegment));
  __detail::local_sort(lsegment, comp);

  /* find splitting values - limits of areas to send to other processes */
  __detail::splitters<valT>(lsegment, comp, vec_split_i, vec_split_s);
  default_comm().alltoall(vec_split_s, vec_rsizes, 1);

  /* prepare data to send and receive */
  std::exclusive_scan(vec_rsizes.begin(), vec_rsizes.end(),
                      vec_rindices.begin(), 0);
  const std::size_t _recv_elems = vec_rindices.back() + vec_rsizes.back();

  /* send and receive data belonging to each node, then redistribute
   * data to achieve size of data equal to size of local segment */
  /* async i_all_gather causes problems on some systems */
  // MPI_Request req_recvelems;
  default_comm().all_gather(_recv_elems, vec_recv_elems);

  /* buffer for received data */
  buffer<valT> vec_recvdata(_recv_elems);

  /* send data not belonging and receive data belonging to local  processes
   */
  default_comm().alltoallv(lsegment, vec_split_s, vec_split_i, vec_recvdata,
                           vec_rsizes, vec_rindices);

  /* TODO: vec recvdata is partially sorted, implementation of merge on GPU is
   * desirable */
  __detail::local_sort(vec_recvdata, comp);
  // MPI_Wait(&req_recvelems, MPI_STATUS_IGNORE);

  _total_elems = std::reduce(vec_recv_elems.begin(), vec_recv_elems.end());

  /* prepare data for shift to neighboring processes */
  std::vector<int64_t> vec_shift(_comm_size - 1);

  const auto desired_elems_num = (_total_elems + _comm_size - 1) / _comm_size;

  vec_shift[0] = desired_elems_num - vec_recv_elems[0];
  for (std::size_t _i = 1; _i < _comm_size - 1; _i++) {
    vec_shift[_i] = vec_shift[_i - 1] + desired_elems_num - vec_recv_elems[_i];
  }

  const int64_t shift_left = _comm_rank == 0 ? 0 : -vec_shift[_comm_rank - 1];
  const int64_t shift_right =
      _comm_rank == _comm_size - 1 ? 0 : vec_shift[_comm_rank];

  buffer<valT> vec_left(std::max(shift_left, 0L));
  buffer<valT> vec_right(std::max(shift_right, 0L));

  /* shift data if necessary, to have exactly the number of elements equal to
   * lsegment size */
  __detail::shift_data<valT>(shift_left, shift_right, vec_recvdata, vec_left,
                             vec_right);

  /* copy results to distributed vector's local segment */
  __detail::copy_results<valT>(lsegment, shift_left, shift_right, vec_recvdata,
                               vec_left, vec_right);
} // __detail::dist_sort

} // namespace __detail

template <dr::distributed_range R, typename Compare = std::less<>>
void sort(R &r, Compare &&comp = Compare()) {

  using valT = typename R::value_type;

  std::size_t _comm_rank = default_comm().rank();
  std::size_t _comm_size = default_comm().size(); // dr-style ignore

  if (_comm_size == 1) {
    DRLOG("mhp::sort() - one node only");
    auto &&lsegment = local_segment(r);
    __detail::local_sort(lsegment, comp);

  } else if (rng::size(r) <= (_comm_size - 1) * (_comm_size - 1)) {
    /* Distributed vector of size <= (comm_size-1) * (comm_size-1) may have
     * 0-size local segments. It is also small enough to prefer sequential sort
     */
    DRLOG("mhp::sort() - local sort on node 0");

    std::vector<valT> vec_recvdata(rng::size(r));
    dr::mhp::copy(0, r, rng::begin(vec_recvdata));

    if (_comm_rank == 0) {
      rng::sort(vec_recvdata, comp);
    }
    dr::mhp::barrier();
    dr::mhp::copy(0, vec_recvdata, rng::begin(r));

  } else {
    DRLOG("mhp::sort() - distributed sort");
    __detail::dist_sort(r, comp);
    dr::mhp::barrier();
  }
}

template <dr::distributed_iterator RandomIt, typename Compare = std::less<>>
void sort(RandomIt first, RandomIt last, Compare comp = Compare()) {
  sort(rng::subrange(first, last), comp);
}

} // namespace dr::mhp
