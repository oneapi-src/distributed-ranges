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
void sort_quick_merge(R &r, std::size_t root = 0, Compare comp = Compare()) {
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

  std::vector<T> vec_lmedians(_comm_size - 1);
  std::vector<T> vec_gmedians((_comm_size - 1) * _comm_size);

  std::size_t _segsize = rng::distance(lsegment);
  std::size_t _step = _segsize / _comm_size;

  /* calculate splitting values and indices - find n-1 "medians" splitting each
   * segment */

  for (std::size_t _i = 0; _i < _comm_size - 1; _i++) {
    vec_lmedians[_i] = lsegment[(_i + 1) * _step];
  }

  default_comm().all_gather(vec_lmedians.data(), vec_gmedians.data(),
                            (_comm_size - 1) * sizeof(T));

  rng::sort(vec_gmedians, comp);

  std::vector<T> split_v(_comm_size);
  _step = rng::size(vec_gmedians) / _comm_size;
  for (std::size_t _i = 0; _i < _comm_size; _i++) {
    split_v[_i] = vec_gmedians[std::size_t((_i + 0.5) * _step)];
  }

  /* calculate splitting indices (start of buffers) and sizes of buffers to send
   */

  std::vector<int> split_i(_comm_size), split_s(_comm_size);

  std::size_t segidx = 0, vidx = 0;
  split_i[vidx++] = 0;

  while (vidx < _comm_size) {
    if (*(lsegment.begin() + segidx) >= split_v[vidx - 1]) {
      split_i[vidx] = segidx * sizeof(T);
      if (vidx)
        split_s[vidx - 1] = split_i[vidx] - split_i[vidx - 1];
      vidx++;
    } else
      segidx++;
  }
  split_s[vidx - 1] = _segsize * sizeof(T) - split_i[vidx - 1];

  /* send data size to each node */

  std::vector<int> rsizes_(_comm_size), rindices_(_comm_size); // recv buffers

  MPI_Alltoall(split_s.data(), sizeof(int), MPI_BYTE, rsizes_.data(),
               sizeof(int), MPI_BYTE, default_comm().mpi_comm());

  rindices_[0] = 0;
  for (int i = 1; i < _comm_size; i++) {
    rindices_[i] = std::reduce(rsizes_.begin(), rsizes_.begin() + i);
  }

  std::size_t recvsum_ = std::reduce(rsizes_.begin(), rsizes_.end());

  /* send and receive data belonging to each node */
  std::vector<T> recvdata(recvsum_ / sizeof(T));

  MPI_Alltoallv(lsegment.data(), split_s.data(), split_i.data(), MPI_BYTE,
                recvdata.data(), rsizes_.data(), rindices_.data(), MPI_BYTE,
                default_comm().mpi_comm());

  rng::sort(recvdata, comp);
  fmt::print("{}: recvdata {} / {}\n", _comm_rank, recvdata,
             rng::size(recvdata));

  // Now redistribute data
  // ...
}

template <dr::distributed_range R, typename Compare = std::less<>>
void sort(R &r, std::size_t root = 0, Compare comp = Compare()) {
  sort_quick_merge(r, root, comp);
}

template <dr::distributed_iterator RandomIt, typename Compare = std::less<>>
void sort(RandomIt first, RandomIt last, std::size_t root = 0,
          Compare comp = Compare()) {
  sort(rng::subrange(first, last), comp);
}

} // namespace dr::mhp
