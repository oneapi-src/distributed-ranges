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
#ifdef SYCL_LANGUAGE_VERSION

    // auto q = dr::shp::__detail::default_queue();
    auto q = sycl::queue();
    auto policy = oneapi::dpl::execution::make_device_policy(q);
    auto &&local_segment = dr::ranges::__detail::local(r);

    oneapi::dpl::sort(policy, rng::begin(local_segment),
                      rng::end(local_segment), comp);

#else
    rng::sort(rng::begin(r), rng::end(r), comp);
#endif
  }
}

template <dr::distributed_range R, typename Compare>
void dist_sort(R &r, Compare &&comp) {
  using valT = typename R::value_type;

  std::size_t _comm_rank = default_comm().rank();
  std::size_t _comm_size = default_comm().size(); // dr-style ignore

  auto &&lsegment = local_segment(r);

  /* sort local segment */

  __detail::local_sort(lsegment, comp);

  std::vector<valT> vec_lmedians(_comm_size - 1);
  std::vector<valT> vec_gmedians((_comm_size - 1) * _comm_size);

  double _step = (double)rng::size(lsegment) / (double)_comm_size;

  /* calculate splitting values and indices - find n-1 dividers splitting each
   * segment into equal parts */

  for (std::size_t _i = 0; _i < rng::size(vec_lmedians); _i++) {
    vec_lmedians[_i] = lsegment[(std::size_t)(_i + 1) * _step];
  }

  default_comm().all_gather(vec_lmedians, vec_gmedians);

  rng::sort(rng::begin(vec_gmedians), rng::end(vec_gmedians), comp);

  std::vector<valT> vec_split_v(_comm_size - 1);
  _step = rng::size(vec_gmedians) / (_comm_size - 1);

  /* find splitting values - medians of dividers */

  for (std::size_t _i = 0; _i < _comm_size - 1; _i++) {
    vec_split_v[_i] = vec_gmedians[std::size_t((_i + 0.5) * _step)];
  }

  /* calculate splitting indices (start of buffers) and sizes of buffers to send
   */

  std::vector<std::size_t> vec_split_i(_comm_size, 0);
  std::vector<std::size_t> vec_split_s(_comm_size, 0);

  std::size_t segidx = 0, vidx = 1;

  while (vidx < _comm_size) {
    assert(segidx < rng::size(lsegment));
    if (comp(vec_split_v[vidx - 1], *(lsegment.begin() + segidx))) {
      vec_split_i[vidx] = segidx;
      vec_split_s[vidx - 1] = vec_split_i[vidx] - vec_split_i[vidx - 1];
      std::size_t _sum = std::reduce(vec_split_s.begin(), vec_split_s.end());
      if (_sum > rng::size(lsegment)) {
        vec_split_s[vidx - 1] -= _sum - rng::size(lsegment);
      }
      vidx++;
    } else {
      segidx++;
      if (segidx >= rng::size(lsegment))
        break;
    }
  }
  vec_split_s[vidx - 1] =
      ((int)(rng::size(lsegment) - vec_split_i[vidx - 1]) > 0)
          ? (rng::size(lsegment) - vec_split_i[vidx - 1])
          : 0;

  assert(vec_split_s[vidx - 1] >= 0);

  /* send data size to each node */

  std::vector<std::size_t> vec_rsizes(_comm_size, 0);
  std::vector<std::size_t> vec_rindices(_comm_size, 0); // recv buffers

  default_comm().alltoall(vec_split_s, vec_rsizes, 1);

  for (std::size_t i = 1; i < _comm_size; i++) {
    vec_rindices[i] = std::reduce(vec_rsizes.begin(), vec_rsizes.begin() + i);
  }
  std::size_t _recvsum = std::reduce(vec_rsizes.begin(), vec_rsizes.end());

  /* send and receive data belonging to each node */

  std::vector<valT> vec_recvdata(_recvsum);
  std::size_t _recv_elems = rng::size(vec_recvdata);

  default_comm().alltoallv(lsegment, vec_split_s, vec_split_i, vec_recvdata,
                           vec_rsizes, vec_rindices);

  __detail::local_sort(vec_recvdata, comp);

  /* Now redistribute data to achieve size of data equal to size of local
   * segment */

  std::vector<std::size_t> vec_recv_elems(_comm_size);

  default_comm().all_gather(_recv_elems, vec_recv_elems);

  std::size_t _total_elems =
      std::reduce(vec_recv_elems.begin(), vec_recv_elems.end());

  std::vector<int> vec_shift(_comm_size - 1);

  vec_shift[0] =
      ((_total_elems + _comm_size - 1) / _comm_size) - vec_recv_elems[0];
  for (std::size_t _i = 1; _i < _comm_size - 1; _i++) {
    vec_shift[_i] = vec_shift[_i - 1] +
                    ((_total_elems + _comm_size - 1) / _comm_size) -
                    vec_recv_elems[_i];
  }

  int shift_left = _comm_rank == 0 ? 0 : -vec_shift[default_comm().prev()];
  int shift_right = _comm_rank == _comm_size - 1 ? 0 : vec_shift[_comm_rank];

  MPI_Request req_l, req_r;
  MPI_Status stat_l, stat_r;
  communicator::tag t = communicator::tag::halo_index;

  std::vector<valT> vec_left(shift_left > 0 ? (shift_left) : 0);
  std::vector<valT> vec_right(shift_right > 0 ? (shift_right) : 0);

  if ((int)rng::size(vec_recvdata) < -shift_left) {

    assert(shift_right > 0);

    default_comm().irecv(vec_right, default_comm().next(), t, &req_r);
    MPI_Wait(&req_r, &stat_r);

    vec_recvdata.insert(vec_recvdata.end(), vec_right.begin(), vec_right.end());
    vec_right.clear();

    default_comm().isend(vec_recvdata.data(), -shift_left,
                         default_comm().prev(), t, &req_l);
    MPI_Wait(&req_l, &stat_l);

  } else if ((int)rng::size(vec_recvdata) < -shift_right) {

    assert(shift_left > 0);
    default_comm().irecv(vec_left, default_comm().prev(), t, &req_l);
    MPI_Wait(&req_l, &stat_l);
    vec_left.insert(vec_left.end(), vec_recvdata.begin(), vec_recvdata.end());
    std::swap(vec_left, vec_recvdata);
    vec_left.clear();

    default_comm().isend((valT *)(vec_recvdata.data()) +
                             rng::size(vec_recvdata) + shift_right,
                         -shift_right, default_comm().next(), t, &req_r);
    MPI_Wait(&req_r, &stat_r);
  } else {

    if (shift_left < 0) {
      default_comm().isend(vec_recvdata.data(), -shift_left,
                           default_comm().prev(), t, &req_l);
    } else if (shift_left > 0) {
      default_comm().irecv(vec_left, default_comm().prev(), t, &req_l);
    }

    if (shift_right > 0) {
      default_comm().irecv(vec_right, default_comm().next(), t, &req_r);
    } else if (shift_right < 0) {
      default_comm().isend((valT *)(vec_recvdata.data()) +
                               rng::size(vec_recvdata) + shift_right,
                           -shift_right, default_comm().next(), t, &req_r);
    }

      if (shift_left != 0)
        MPI_Wait(&req_l, &stat_l);
      if (shift_right != 0)
        MPI_Wait(&req_r, &stat_r);
  }

  std::size_t invalidate_left = (shift_left < 0) ? -shift_left : 0;
  std::size_t invalidate_right = (shift_right < 0) ? -shift_right : 0;
  std::size_t pos = 0;

  if (rng::size(vec_left) > 0) {
    rng::copy(vec_left, lsegment.begin());
    pos = rng::size(vec_left);
  }

  rng::copy(vec_recvdata.begin() + invalidate_left,
            vec_recvdata.end() - invalidate_right, lsegment.begin() + pos);

  pos += rng::size(vec_recvdata) - (invalidate_right + invalidate_left);

  if (rng::size(vec_right) > 0) {
    rng::copy(vec_right, lsegment.begin() + pos);
  }
}

} // namespace __detail

template <dr::distributed_range R, typename Compare = std::less<>>
void sort(R &r, Compare &&comp = Compare()) {

  using valT = typename R::value_type;

  std::size_t _comm_rank = default_comm().rank();
  std::size_t _comm_size = default_comm().size(); // dr-style ignore

  auto &&lsegment = local_segment(r);

  if (_comm_size == 0)
    return;
  else if (_comm_size == 1) {
    __detail::local_sort(lsegment, comp);

  } else if (rng::size(r) <= (_comm_size - 1) * (_comm_size - 1)) {
    /* Distributed vector of size <= (comm_size-1) * (comm_size-1) may have
     * 0-size local segments. It is also small enough to prefer sequential sort
     */

    std::vector<valT> vec_recvdata(rng::size(r));
    dr::mhp::copy(0, r, rng::begin(vec_recvdata));

    if (_comm_rank == 0) {
      rng::sort(vec_recvdata, comp);
    }

    dr::mhp::barrier();
    dr::mhp::copy(0, vec_recvdata, rng::begin(r));

  } else {
    __detail::dist_sort(r, comp);
  }
}

template <dr::distributed_iterator RandomIt, typename Compare = std::less<>>
void sort(RandomIt first, RandomIt last, Compare comp = Compare()) {
  sort(rng::subrange(first, last), comp);
}

} // namespace dr::mhp
