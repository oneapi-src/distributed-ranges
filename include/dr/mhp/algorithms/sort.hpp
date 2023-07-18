// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <mpi.h>

#include <algorithm>
#include <utility>

#include <oneapi/dpl/algorithm>

#include <dr/concepts/concepts.hpp>
#include <dr/detail/logger.hpp>
#include <dr/detail/onedpl_direct_iterator.hpp>
#include <dr/detail/ranges_shim.hpp>
#include <dr/mhp/global.hpp>

namespace dr::mhp {

template <dr::distributed_range R, typename Compare = std::less<>>
void sort_quick_merge(R &r, Compare comp = Compare()) {
  using T = R::value_type;
  auto &&segments = dr::ranges::segments(r);

  if (rng::size(segments) == 0)
    return;
  else if (rng::size(segments) == 1) {
    rng::sort(segments[0], comp);
    return;
  }

  // quicksort and merge

  std::size_t _comm_rank = default_comm().rank();
  std::size_t _comm_size = default_comm().size(); // dr-style ignore

  /* sort local segment */

  auto &&lsegment = local_segment(r);

  rng::sort(lsegment, comp);

  fmt::print("{}: lsegment {}\n", _comm_rank, lsegment);
  dr::mhp::barrier();

  std::vector<T> vec_lmedians(_comm_size - 1);
  std::vector<T> vec_gmedians((_comm_size - 1) * _comm_size);

  std::size_t _segsize = rng::distance(lsegment);
  std::size_t _step = _segsize / _comm_size;

  /* calculate splitting values and indices - find n-1 "medians" splitting each
   * segment */

  for (std::size_t _i = 0; _i < _comm_size - 1; _i++) {
    vec_lmedians[_i] = lsegment[(_i + 1) * _step];
  }

  // fmt::print("{}: vec_lmedians {}\n", _comm_rank, vec_lmedians);

  default_comm().all_gather(vec_lmedians.data(), vec_gmedians.data(),
                            (_comm_size - 1) * sizeof(T));

  // fmt::print("{}: vec_gmedians (u) {}\n", _comm_rank, vec_gmedians);

  rng::sort(vec_gmedians, comp);

  // fmt::print("{}: vec_gmedians (s) {}\n", _comm_rank, vec_gmedians);

  std::vector<T> vec_split_v(_comm_size - 1);
  _step = rng::size(vec_gmedians) / (_comm_size - 1);

  for (std::size_t _i = 0; _i < _comm_size - 1; _i++) {
    vec_split_v[_i] = vec_gmedians[std::size_t((_i + 0.5) * _step)];
  }

  fmt::print("{}: vec_split_v {}\n", _comm_rank, vec_split_v);

  /* calculate splitting indices (start of buffers) and sizes of buffers to send
   */

  std::vector<int> vec_split_i(_comm_size), vec_split_s(_comm_size);

  std::size_t segidx = 0, vidx = 0;
  vec_split_i[vidx++] = 0;

  while (vidx < _comm_size) {
    if (*(lsegment.begin() + segidx) >= vec_split_v[vidx - 1]) {
      vec_split_i[vidx] = segidx * sizeof(T);
      vec_split_s[vidx - 1] = vec_split_i[vidx] - vec_split_i[vidx - 1];
      vidx++;
    } else
      segidx++;
  }
  vec_split_s[vidx - 1] = _segsize * sizeof(T) - vec_split_i[vidx - 1];

  fmt::print("{}: vec_split_i {}\n", _comm_rank, vec_split_i);
  fmt::print("{}: vec_split_s {}\n", _comm_rank, vec_split_s);

  /* send data size to each node */

  std::vector<int> vec_rsizes(_comm_size);
  std::vector<int> vec_rindices(_comm_size); // recv buffers

  MPI_Alltoall(vec_split_s.data(), sizeof(int), MPI_BYTE, vec_rsizes.data(),
               sizeof(int), MPI_BYTE, default_comm().mpi_comm());

  vec_rindices[0] = 0;
  for (int i = 1; i < _comm_size; i++) {
    vec_rindices[i] = std::reduce(vec_rsizes.begin(), vec_rsizes.begin() + i);
  }
  std::size_t _recvsum = std::reduce(vec_rsizes.begin(), vec_rsizes.end());

  fmt::print("{}: vec_rsizes   {} / {}\n", _comm_rank, vec_rsizes, _recvsum);
  fmt::print("{}: vec_rindices {}\n", _comm_rank, vec_rindices);

  /* send and receive data belonging to each node */
  std::vector<T> vec_recvdata(_recvsum / sizeof(T));
  std::size_t _recv_elems = rng::size(vec_recvdata);

  MPI_Alltoallv(lsegment.data(), vec_split_s.data(), vec_split_i.data(),
                MPI_BYTE, vec_recvdata.data(), vec_rsizes.data(),
                vec_rindices.data(), MPI_BYTE, default_comm().mpi_comm());

  fmt::print("{}: recvdata (u) {} / {}\n", _comm_rank, vec_recvdata,
             _recv_elems);

  rng::sort(vec_recvdata, comp);

  fmt::print("{}: recvdata (s) {} / {}\n", _comm_rank, vec_recvdata,
             _recv_elems);
  dr::mhp::barrier();

  // Now redistribute data

  std::vector<std::size_t> vec_recv_elems(_comm_size);

  default_comm().all_gather(_recv_elems, vec_recv_elems);

  std::size_t _total_elems =
      std::reduce(vec_recv_elems.begin(), vec_recv_elems.end());

  // fmt::print("{}: vec_recv_elems {}\n", _comm_rank, vec_recv_elems);

  std::vector<std::pair<int, int>> vec_shifts(_comm_size);

  vec_shifts[0].first = 0;

  for (int _i = 0; _i < _comm_size - 1; _i++) {
    vec_shifts[_i].second = -vec_shifts[_i].first +
                            ((_total_elems + _comm_size - 1) / _comm_size) -
                            vec_recv_elems[_i];
    vec_shifts[_i + 1].first = -vec_shifts[_i].second;
  }

  vec_shifts[_comm_size - 1].second = 0;

  fmt::print("{}: vec_shifts[", _comm_rank);
  for (int _i = 0; _i < _comm_size; _i++)
    fmt::print("[{},{}] ", vec_shifts[_i].first, vec_shifts[_i].second);
  fmt::print("\n");

  MPI_Request req_l, req_r;
  MPI_Status stat_l, stat_r;
  communicator::tag t = communicator::tag::halo_index;

  std::vector<T> vec_left(
      vec_shifts[_comm_rank].first > 0 ? (vec_shifts[_comm_rank].first) : 0);
  std::vector<T> vec_right(
      vec_shifts[_comm_rank].second > 0 ? (vec_shifts[_comm_rank].second) : 0);

  /* left-hand (lower rank) redistribution */

  if (vec_shifts[_comm_rank].first < 0) {
    // fmt::print("{}: send left {}\n", _comm_rank,
    // -vec_shifts[_comm_rank].first);
    default_comm().isend(vec_recvdata.data(),
                         -vec_shifts[_comm_rank].first * sizeof(T),
                         _comm_rank - 1, t, &req_l);
    MPI_Wait(&req_l, &stat_l);

  } else if (vec_shifts[_comm_rank].first > 0) {
    // fmt::print("{}: recv left {}\n", _comm_rank,
    // vec_shifts[_comm_rank].first);
    default_comm().irecv(vec_left.data(),
                         vec_shifts[_comm_rank].first * sizeof(T),
                         _comm_rank - 1, t, &req_l);
    MPI_Wait(&req_l, &stat_l);
  }

  /* right-hand (higher rank) redistribution */
  if (vec_shifts[_comm_rank].second > 0) {
    // fmt::print("{}: recv right {}\n", _comm_rank,
    //            vec_shifts[_comm_rank].second);
    default_comm().irecv(vec_right.data(),
                         vec_shifts[_comm_rank].second * sizeof(T),
                         _comm_rank + 1, t, &req_r);
    MPI_Wait(&req_r, &stat_r);

  } else if (vec_shifts[_comm_rank].second < 0) {

    // fmt::print("{}: send right {}\n", _comm_rank,
    //            vec_shifts[_comm_rank].second);

    default_comm().isend((T *)(vec_recvdata.data()) + _recv_elems +
                             vec_shifts[_comm_rank].second,
                         vec_shifts[_comm_rank].second * sizeof(T),
                         _comm_rank + 1, t, &req_r);
    MPI_Wait(&req_r, &stat_r);
  }

  fmt::print("{}: vec_left {} size {} vec_right {} size {}\n", _comm_rank,
             vec_left, rng::size(vec_left), vec_right, rng::size(vec_right));

  std::size_t invalidate_left =
      (vec_shifts[_comm_rank].first < 0) ? -vec_shifts[_comm_rank].first : 0;

  std::size_t invalidate_right =
      (vec_shifts[_comm_rank].second < 0) ? -vec_shifts[_comm_rank].second : 0;

  std::size_t pos = 0;

  dr::mhp::barrier();

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

  fmt::print("{}: lsegment (2) {}\n", _comm_rank, lsegment);

  if (_comm_rank == 0)
    fmt::print("RESULT {}\n", r);
}

template <dr::distributed_range R, typename Compare = std::less<>>
void sort(R &r, Compare comp = Compare()) {
  sort_quick_merge(r, comp);
}

template <dr::distributed_iterator RandomIt, typename Compare = std::less<>>
void sort(RandomIt first, RandomIt last, Compare comp = Compare()) {
  sort(rng::subrange(first, last), comp);
}

} // namespace dr::mhp
