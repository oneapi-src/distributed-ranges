// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <mpi.h>

#include <algorithm>
#include <utility>

#ifdef SYCL_LANGUAGE_VERSION
#include <oneapi/dpl/algorithm>
#endif

#include <dr/concepts/concepts.hpp>
#include <dr/detail/logger.hpp>
#include <dr/detail/onedpl_direct_iterator.hpp>
#include <dr/detail/ranges_shim.hpp>
#include <dr/mhp/global.hpp>

namespace dr::mhp {

template <dr::distributed_range R, typename Compare = std::less<>>
void sort(R &r, Compare comp = Compare()) {
  using T = R::value_type;

  std::size_t _comm_rank = default_comm().rank();
  std::size_t _comm_size = default_comm().size(); // dr-style ignore

  auto &&lsegment = local_segment(r);
  assert(rng::size(lsegment) > 0);

  if (_comm_size == 0)
    return;
  else if (_comm_size == 1) {
#ifdef SYCL_LANGUAGE_VERSION
    oneapi::dpl::sort(oneapi::dpl::execution::dpcpp_default, lsegment.begin(),
                      lsegment.end(), comp);
#else
    rng::sort(lsegment, comp);
#endif
    return;
  }

  /* sort local segment */

#ifdef SYCL_LANGUAGE_VERSION
  oneapi::dpl::sort(oneapi::dpl::execution::dpcpp_default, lsegment.begin(),
                    lsegment.end(), comp);
#else
  rng::sort(lsegment, comp);
#endif

  std::vector<T> vec_lmedians(_comm_size - 1);
  std::vector<T> vec_gmedians((_comm_size - 1) * _comm_size);

  std::size_t _step = rng::size(lsegment) / _comm_size;

  /* calculate splitting values and indices - find n-1 dividers splitting each
   * segment into equal parts */

  for (std::size_t _i = 0; _i < rng::size(vec_lmedians); _i++) {
    vec_lmedians[_i] = lsegment[(_i + 1) * _step];
  }

  default_comm().all_gather(vec_lmedians, vec_gmedians);

  rng::sort(vec_gmedians, comp);

  std::vector<T> vec_split_v(_comm_size - 1);
  _step = rng::size(vec_gmedians) / (_comm_size - 1);

  /* find splitting values - medians of dividers */
  for (std::size_t _i = 0; _i < _comm_size - 1; _i++) {
    vec_split_v[_i] = vec_gmedians[std::size_t((_i + 0.5) * _step)];
  }

  /* calculate splitting indices (start of buffers) and sizes of buffers to send
   */

  std::vector<int> vec_split_i(_comm_size, 0), vec_split_s(_comm_size, 0);

  std::size_t segidx = 0, vidx = 1;

  while (vidx < _comm_size) {
    if (comp(vec_split_v[vidx - 1], *(lsegment.begin() + segidx))) {
      vec_split_i[vidx] = segidx;
      vec_split_s[vidx - 1] = vec_split_i[vidx] - vec_split_i[vidx - 1];
      std::size_t _sum = std::reduce(vec_split_s.begin(), vec_split_s.end());
      if (_sum > rng::size(lsegment)) {
        vec_split_s[vidx - 1] -= _sum - rng::size(lsegment);
      }
      vidx++;
    } else
      segidx++;
  }
  vec_split_s[vidx - 1] =
      ((int)(rng::size(lsegment) - vec_split_i[vidx - 1]) > 0)
          ? (rng::size(lsegment) - vec_split_i[vidx - 1])
          : 0;

  assert(vec_split_s[vidx - 1] >= 0);

  /* send data size to each node */

  std::vector<int> vec_rsizes(_comm_size, 0);
  std::vector<int> vec_rindices(_comm_size, 0); // recv buffers

  default_comm().alltoall(vec_split_s, vec_rsizes, 1);

  for (std::size_t i = 1; i < _comm_size; i++) {
    vec_rindices[i] = std::reduce(vec_rsizes.begin(), vec_rsizes.begin() + i);
  }
  std::size_t _recvsum = std::reduce(vec_rsizes.begin(), vec_rsizes.end());

  /* send and receive data belonging to each node */

  std::vector<T> vec_recvdata(_recvsum);
  std::size_t _recv_elems = rng::size(vec_recvdata);

  default_comm().alltoallv(lsegment.data(), vec_split_s, vec_split_i,
                           vec_recvdata.data(), vec_rsizes, vec_rindices);

  rng::sort(vec_recvdata, comp);

  /* Now redistribute data to achievesize of data in every segment equal to size
   * of local segment */

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

  std::vector<T> vec_left(shift_left > 0 ? (shift_left) : 0);
  std::vector<T> vec_right(shift_right > 0 ? (shift_right) : 0);

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

    default_comm().isend((T *)(vec_recvdata.data()) + _recv_elems + shift_right,
                         -shift_right, default_comm().next(), t, &req_r);
    MPI_Wait(&req_r, &stat_r);
  } else {

    /* left-hand (lower rank) redistribution */
    if (shift_left < 0) {
      default_comm().isend(vec_recvdata.data(), -shift_left,
                           default_comm().prev(), t, &req_l);
      MPI_Wait(&req_l, &stat_l);
    } else if (shift_left > 0) {
      default_comm().irecv(vec_left, default_comm().prev(), t, &req_l);
      MPI_Wait(&req_l, &stat_l);
    }

    /* right-hand (higher rank) redistribution */
    if (shift_right > 0) {
      default_comm().irecv(vec_right, default_comm().next(), t, &req_r);
      MPI_Wait(&req_r, &stat_r);
    } else if (shift_right < 0) {
      default_comm().isend((T *)(vec_recvdata.data()) + _recv_elems +
                               shift_right,
                           -shift_right, default_comm().next(), t, &req_r);
      MPI_Wait(&req_r, &stat_r);
    }
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

template <dr::distributed_iterator RandomIt, typename Compare = std::less<>>
void sort(RandomIt first, RandomIt last, Compare comp = Compare()) {
  sort(rng::subrange(first, last), comp);
}

} // namespace dr::mhp
