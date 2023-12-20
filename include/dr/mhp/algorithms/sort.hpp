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

template <rng::forward_range R, typename Compare>
void local_sort(R &r, Compare &&comp) {
  if (rng::size(r) >= 2) {
    if (mhp::use_sycl()) {
#ifdef SYCL_LANGUAGE_VERSION
      auto policy = dpl_policy();
      auto &&local_segment = dr::ranges::__detail::local(r);
      drlog.debug("GPU dpl::sort(), size {}\n", rng::size(r));
      oneapi::dpl::sort(
          policy, dr::__detail::direct_iterator(rng::begin(local_segment)),
          dr::__detail::direct_iterator(rng::end(local_segment)), comp);
#else
      assert(false);
#endif
    } else {
      drlog.debug("cpu rng::sort, size {}\n", rng::size(r));
      rng::sort(rng::begin(r), rng::end(r), comp);
    }
  }
}

/* elements of dist_sort */
template <typename valT, typename Compare, typename Seg>
void splitters(Seg &lsegment, Compare &&comp,
               std::vector<std::size_t> &vec_split_i,
               std::vector<std::size_t> &vec_split_s) {

  const std::size_t _comm_size = default_comm().size(); // dr-style ignore

  assert(rng::size(vec_split_i) == _comm_size);
  assert(rng::size(vec_split_s) == _comm_size);

  std::vector<valT> vec_lmedians(_comm_size + 1);
  std::vector<valT> vec_gmedians((_comm_size + 1) * _comm_size);

  const double _step_m = static_cast<double>(rng::size(lsegment)) /
                         static_cast<double>(_comm_size);

  /* calculate splitting values and indices - find n-1 dividers splitting
   * each segment into equal parts */

  for (std::size_t _i = 0; _i < rng::size(vec_lmedians); _i++) {
    vec_lmedians[_i] = lsegment[_i * _step_m];
  }
  vec_lmedians.back() = lsegment.back();

  default_comm().all_gather(vec_lmedians, vec_gmedians);

  rng::sort(rng::begin(vec_gmedians), rng::end(vec_gmedians), comp);

  std::vector<valT> vec_split_v(_comm_size - 1);

  for (std::size_t _i = 0; _i < _comm_size - 1; _i++) {
    vec_split_v[_i] = vec_gmedians[(_i + 1) * (_comm_size + 1) - 1];
  }

  std::size_t segidx = 0, vidx = 1;

  while (vidx < _comm_size && segidx < rng::size(lsegment)) {
    if (comp(vec_split_v[vidx - 1], *(lsegment.begin() + segidx))) {
      vec_split_i[vidx] = segidx;
      vec_split_s[vidx - 1] = vec_split_i[vidx] - vec_split_i[vidx - 1];
      vidx++;
    } else {
      segidx++;
    }
  }
  assert(rng::size(lsegment) > vec_split_i[vidx - 1]);
  vec_split_s[vidx - 1] = rng::size(lsegment) - vec_split_i[vidx - 1];
}

template <typename valT, typename Alloc>
void shift_data(const int shift_left, const int shift_right,
                std::vector<valT, Alloc> &vec_recvdata,
                std::vector<valT, Alloc> &vec_left,
                std::vector<valT, Alloc> &vec_right) {

  const std::size_t _comm_rank = default_comm().rank();

  MPI_Request req_l, req_r;
  MPI_Status stat_l, stat_r;
  const communicator::tag t = communicator::tag::halo_index;

  if (static_cast<int>(rng::size(vec_recvdata)) < -shift_left) {
    // Too little data in recv buffer to shift left - first get from right, then
    // send left
    drlog.debug("Get from right first, recvdata size {} shl {}\n",
                rng::size(vec_recvdata), -shift_left);
    // ** This will never happen, because values eq to split go left **
    assert(false);
  } else if (static_cast<int>(rng::size(vec_recvdata)) < -shift_right) {
    // Too little data in buffer to shift right - first get from left, then send
    // right
    assert(shift_left > 0);
    default_comm().irecv(vec_left, _comm_rank - 1, t, &req_l);
    MPI_Wait(&req_l, &stat_l);

    vec_left.insert(vec_left.end(), vec_recvdata.begin(), vec_recvdata.end());
    std::swap(vec_left, vec_recvdata);
    vec_left.clear();

    default_comm().isend((valT *)(vec_recvdata.data()) +
                             rng::size(vec_recvdata) + shift_right,
                         -shift_right, _comm_rank + 1, t, &req_r);
    MPI_Wait(&req_r, &stat_r);
  } else {
    // enough data in recv buffer

    if (shift_left < 0) {
      default_comm().isend(vec_recvdata.data(), -shift_left, _comm_rank - 1, t,
                           &req_l);
    } else if (shift_left > 0) {
      default_comm().irecv(vec_left, _comm_rank - 1, t, &req_l);
    }

    if (shift_right > 0) {
      default_comm().irecv(vec_right, _comm_rank + 1, t, &req_r);
    } else if (shift_right < 0) {
      default_comm().isend((valT *)(vec_recvdata.data()) +
                               rng::size(vec_recvdata) + shift_right,
                           -shift_right, _comm_rank + 1, t, &req_r);
    }

    if (shift_left != 0)
      MPI_Wait(&req_l, &stat_l);
    if (shift_right != 0)
      MPI_Wait(&req_r, &stat_r);
  }
}

template <typename valT, typename Alloc>
void copy_results(auto &lsegment, const int shift_left, const int shift_right,
                  std::vector<valT, Alloc> &vec_recvdata,
                  std::vector<valT, Alloc> &vec_left,
                  std::vector<valT, Alloc> &vec_right) {
  const std::size_t invalidate_left = std::max(-shift_left, 0);
  const std::size_t invalidate_right = std::max(-shift_right, 0);

  const std::size_t size_l = rng::size(vec_left);
  const std::size_t size_r = rng::size(vec_right);
  const std::size_t size_d =
      rng::size(vec_recvdata) - (invalidate_left + invalidate_right);

  if (mhp::use_sycl()) {
#ifdef SYCL_LANGUAGE_VERSION
    sycl::event e_l, e_d, e_r;

    if (size_l > 0)
      e_l = sycl_queue().copy(vec_left.data(), lsegment.data(), size_l);
    if (size_r > 0)
      e_r = sycl_queue().copy(vec_right.data(),
                              lsegment.data() + size_l + size_d, size_r);
    e_d = sycl_queue().copy(vec_recvdata.data() + invalidate_left,
                            lsegment.data() + size_l, size_d);
    if (size_l > 0)
      e_l.wait();
    if (size_r > 0)
      e_r.wait();
    e_d.wait();

#else
    assert(false);
#endif
  } else {
    if (size_l > 0)
      std::memcpy(lsegment.data(), vec_left.data(), size_l * sizeof(valT));
    if (size_r > 0)
      std::memcpy(lsegment.data() + size_l + size_d, vec_right.data(),
                  size_r * sizeof(valT));

    std::memcpy(lsegment.data() + size_l, vec_recvdata.data() + invalidate_left,
                size_d * sizeof(valT));
  }
}

template <dr::distributed_range R, typename Compare>
void dist_sort(R &r, Compare &&comp) {

  using valT = typename R::value_type;

  const std::size_t _comm_rank = default_comm().rank();
  const std::size_t _comm_size = default_comm().size(); // dr-style ignore

#ifdef SYCL_LANGUAGE_VERSION
  auto policy = dpl_policy();
  sycl::usm_allocator<valT, sycl::usm::alloc::shared> alloc(policy.queue());
#endif

  auto &&lsegment = local_segment(r);

  std::vector<std::size_t> vec_split_i(_comm_size, 0);
  std::vector<std::size_t> vec_split_s(_comm_size, 0);
  std::vector<std::size_t> vec_rsizes(_comm_size, 0);
  std::vector<std::size_t> vec_rindices(_comm_size, 0);
  std::vector<std::size_t> vec_recv_elems(_comm_size, 0);
  std::size_t _total_elems = 0;

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

  MPI_Request req_recvelems;
  default_comm().i_all_gather(_recv_elems, vec_recv_elems, &req_recvelems);

  /* buffer for received data */
#ifdef SYCL_LANGUAGE_VERSION
  std::vector<valT, decltype(alloc)> vec_recvdata(_recv_elems, alloc);
#else
  std::vector<valT> vec_recvdata(_recv_elems);
#endif

  /* send data not belonging and receive data belonging to local  processes
   */
  default_comm().alltoallv(lsegment, vec_split_s, vec_split_i, vec_recvdata,
                           vec_rsizes, vec_rindices);

  /* TODO: vec recvdata is partially sorted, implementation of merge on GPU is
   * desirable */
  __detail::local_sort(vec_recvdata, comp);

  MPI_Wait(&req_recvelems, MPI_STATUS_IGNORE);

  _total_elems = std::reduce(vec_recv_elems.begin(), vec_recv_elems.end());

  /* prepare data for shift to neighboring processes */
  std::vector<int> vec_shift(_comm_size - 1);

  const auto desired_elems_num = (_total_elems + _comm_size - 1) / _comm_size;

  vec_shift[0] = desired_elems_num - vec_recv_elems[0];
  for (std::size_t _i = 1; _i < _comm_size - 1; _i++) {
    vec_shift[_i] = vec_shift[_i - 1] + desired_elems_num - vec_recv_elems[_i];
  }

  const int shift_left = _comm_rank == 0 ? 0 : -vec_shift[_comm_rank - 1];
  const int shift_right =
      _comm_rank == _comm_size - 1 ? 0 : vec_shift[_comm_rank];

#ifdef SYCL_LANGUAGE_VERSION
  std::vector<valT, decltype(alloc)> vec_left(std::max(shift_left, 0), alloc);
  std::vector<valT, decltype(alloc)> vec_right(std::max(shift_right, 0), alloc);
#else
  std::vector<valT> vec_left(std::max(shift_left, 0));
  std::vector<valT> vec_right(std::max(shift_right, 0));
#endif

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
    drlog.debug("mhp::sort() - single node\n");

    auto &&lsegment = local_segment(r);
    __detail::local_sort(lsegment, comp);

  } else if (rng::size(r) <= (_comm_size - 1) * (_comm_size - 1)) {
    /* Distributed vector of size <= (comm_size-1) * (comm_size-1) may have
     * 0-size local segments. It is also small enough to prefer sequential sort
     */
    drlog.debug("mhp::sort() - local sort\n");

    std::vector<valT> vec_recvdata(rng::size(r));
    dr::mhp::copy(0, r, rng::begin(vec_recvdata));

    if (_comm_rank == 0) {
      rng::sort(vec_recvdata, comp);
    }
    dr::mhp::barrier();
    dr::mhp::copy(0, vec_recvdata, rng::begin(r));

  } else {
    drlog.debug("mhp::sort() - dist sort\n");
    __detail::dist_sort(r, comp);
    dr::mhp::barrier();
  }
}

template <dr::distributed_iterator RandomIt, typename Compare = std::less<>>
void sort(RandomIt first, RandomIt last, Compare comp = Compare()) {
  sort(rng::subrange(first, last), comp);
}

} // namespace dr::mhp
