// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

namespace lib {

// Utilities for zip ranges

//
//
// fill
//
//

/// Collective fill on iterator/sentinel for a distributed range
template <mpi_distributed_contiguous_iterator DI>
void fill(DI first, DI last, auto value) {
  std::fill(first.local(), last.local(), value);
  first.container().comm().barrier();
}

/// Collective fill on distributed range
void fill(mpi_distributed_contiguous_range auto &&r, auto value) {
  lib::fill(r.begin(), r.end(), value);
}

//
//
// for_each
//
//

/// Collective for_each on iterator/sentinel for a distributed range
template <mpi_distributed_contiguous_iterator DI>
void for_each(DI first, DI last, auto op) {
  std::for_each(first.local(), last.local(), op);
  first.container().comm().barrier();
}

/// Collective for_each on distributed range
void for_each(mpi_distributed_contiguous_range auto &&r, auto op) {
  lib::for_each(r.begin(), r.end(), op);
}

/// Collective for_each on zipped distributed range
void for_each(auto &&zr, auto op) {
  const auto &comm = zip_range_comm(zr);
  if (zip_range_conformant(zr)) {
    rng::for_each(zr | lib::local_zip_span(), op);
    comm.barrier();
  } else {
    if (comm.rank() == 0) {
      rng::for_each(zr, op);
    }
    zip_range_fence(zr);
  }
}

//
//
// iota
//
//

/// Collective iota on iterator/sentinel for a distributed range
template <mpi_distributed_contiguous_iterator DI>
void iota(DI first, DI last, auto value) {
  auto &container = first.container();
  if (container.comm().rank() == 0) {
    std::iota(first, last, value);
  }
  container.fence();
}

/// Collective iota on distributed range
void iota(mpi_distributed_contiguous_range auto &&r, auto value) {
  lib::iota(r.begin(), r.end(), value);
}

//
//
// Reduce
//
//

/// Collective reduction on iterator/sentinel for a distributed range
template <mpi_distributed_contiguous_iterator DI, typename T>
T reduce(int root, DI first, DI last, T init, auto &&binary_op) {
  auto val = std::reduce(first.local(), last.local(), 0, binary_op);
  drlog.debug("local reduce: {}\n", val);

  // Gather segment values on root and reduce for final value
  std::vector<T> vals;
  const communicator &comm = first.container().comm();
  comm.gather(val, vals, root);
  if (comm.rank() == root) {
    auto gval = std::reduce(vals.begin(), vals.end(), init, binary_op);
    drlog.debug("global reduce: {}\n", gval);
    return gval;
  } else {
    return 0;
  }
}

/// Collective reduction on a distributed range
template <typename T>
T reduce(int root, mpi_distributed_contiguous_range auto &&r, T init,
         auto &&binary_op) {
  return lib::reduce(root, r.begin(), r.end(), init, binary_op);
}

//
//
// Copy
//
//

/// Collective copy from distributed iterator to distributed iterator
template <mpi_distributed_contiguous_iterator DI>
void copy(DI first, DI last, mpi_distributed_contiguous_iterator auto result) {
  if (first.conforms(result)) {
    std::copy(first.local(), last.local(), result.local());
    result.container().comm().barrier();
  } else {
    if (first.container().comm().rank() == 0) {
      std::copy(first, last, result);
    }
    result.container().fence();
  }
}

/// Collective copy from distributed range to distributed iterator
void copy(mpi_distributed_contiguous_range auto &&r,
          mpi_distributed_contiguous_iterator auto result) {
  lib::copy(r.begin(), r.end(), result);
}

namespace {

auto scatter_data_dst_size(mpi_distributed_contiguous_iterator auto first,
                           std::size_t size, int rank) {
  const auto last = first + size;
  const std::size_t element_size = sizeof(std::iter_value_t<decltype(first)>);
  const auto b = first.remote_offset(rank);
  const auto e = last.remote_offset(rank);
  return (e - b) * element_size;
}

auto scatter_data(mpi_distributed_contiguous_iterator auto first,
                  std::size_t size, std::vector<int> &counts,
                  std::vector<int> &offsets) {
  assert(size > 0);

  const communicator &comm = first.container().comm();
  counts.resize(comm.size());
  offsets.resize(comm.size());

  std::size_t offset = 0;
  for (int rank = 0; rank < comm.size(); rank++) {
    const auto size_on_rank = scatter_data_dst_size(first, size, rank);
    counts[rank] = size_on_rank;
    offsets[rank] = offset;
    offset += size_on_rank;
  }

  drlog.debug("scatter data:\n  counts: {}\n  offsets: {}\n", counts, offsets);
}

inline void *it2raw(std::nullptr_t) { return NULL; }
template <class It> inline void *it2raw(It it) { return &*it; }

} // unnamed namespace

/// Collective copy from local begin/end to distributed iterator.
/// On the non-root rank the `first` parameter is ignored and may be a
/// `nullptr`. Size of the source needs to be provided on all ranks.
void copy(int root, contiguous_iterator_or_nullptr auto first, std::size_t size,
          mpi_distributed_contiguous_iterator auto result) {
  if (size == 0)
    return;

  const communicator &comm = result.container().comm();
  if (root != comm.rank()) {
    comm.scatterv(nullptr, nullptr, nullptr, &*result.local(),
                  scatter_data_dst_size(result, size, comm.rank()), root);
  } else if constexpr (!std::is_same_v<decltype(first), std::nullptr_t>) {
    std::vector<int> counts(comm.size()), offsets(comm.size());
    scatter_data(result, size, counts, offsets);
    comm.scatterv(&*first, counts.data(), offsets.data(), &*result.local(),
                  counts[comm.rank()], root);
  } else {
    assert(false); // nullptr can not be used on root rank
  }
}

/// Collective copy from local begin/end to distributed iterator.
/// On the non-root rank the `first` and `last` parameters are ignored and may
/// be a `nullptr`.
template <contiguous_iterator_or_nullptr IN>
void copy(int root, IN first, IN last,
          mpi_distributed_contiguous_iterator auto result) {
  const communicator &comm = result.container().comm();
  std::size_t size;
  if constexpr (!std::is_same_v<IN, std::nullptr_t>)
    if (root == comm.rank())
      size = std::distance(first, last);

  comm.bcast(&size, sizeof(size), root);
  lib::copy(root, first, size, result);
}

/// Collective copy from local range to distributed iterator
void copy(int root, rng::contiguous_range auto &&r,
          mpi_distributed_contiguous_iterator auto result) {
  lib::copy(root, r.begin(), r.end(), result);
}

/// Collective copy from distributed begin/end to local iterator.
/// On the non-root rank the `result` parameter is ignored and may be a
/// `nullptr`.
template <mpi_distributed_contiguous_iterator DI>
void copy(int root, DI first, DI last,
          contiguous_iterator_or_nullptr auto result) {
  if (last - first == 0)
    return;
  const communicator &comm = first.container().comm();
  std::vector<int> counts(comm.size()), offsets(comm.size());
  assert((not std::is_same_v<decltype(result), std::nullptr_t> ||
          root != comm.rank()));
  scatter_data(first, last - first, counts, offsets);
  comm.gatherv(&*first.local(), counts.data(), offsets.data(), it2raw(result),
               root);
}

/// Collective copy from distributed begin/end to local iterator.
/// On the non-root rank the `result` parameter is ignored and may be a
/// `nullptr`.
void copy(int root, mpi_distributed_contiguous_iterator auto first,
          std::size_t size, contiguous_iterator_or_nullptr auto result) {
  lib::copy(root, first, first + size, result);
}

/// Collective copy from distributed range to local iterator.
/// On the non-root rank the `result` parameter is ignored and may be a
/// `nullptr`.
void copy(int root, mpi_distributed_contiguous_range auto &&r,
          contiguous_iterator_or_nullptr auto result) {
  lib::copy(root, r.begin(), r.end(), result);
}

//
//
// Transform
//
//

/// Collective transform on an iterator/sentinel for a distributed
/// range: 1 in, 1 out
template <mpi_distributed_contiguous_iterator DI>
auto transform(DI first, DI last,
               mpi_distributed_contiguous_iterator auto result, auto op) {
  auto &input = first.container();

  input.halo().exchange_begin();
  input.halo().exchange_finalize();
  if (first.conforms(result)) {
    rng::transform(first.local(), last.local(), result.local(), op);
    input.comm().barrier();
  } else {
    if (input.comm().rank() == 0) {
      rng::transform(first, last, result, op);
    }
  }
  return result + (last - first);
}

/// Collective transform on a distributed range: 1 in, 1 out
auto transform(mpi_distributed_contiguous_range auto &&r,
               mpi_distributed_contiguous_iterator auto result, auto op) {
  return lib::transform(r.begin(), r.end(), result, op);
}

/// Collective transform on an iterator/sentinel for a distributed
/// range: 2 in, 1 out
template <mpi_distributed_contiguous_iterator DI>
auto transform(DI first1, DI last1,
               mpi_distributed_contiguous_iterator auto first2,
               mpi_distributed_contiguous_iterator auto result, auto op) {
  auto &input1 = first1.container();
  auto &input2 = first2.container();

  input1.halo().exchange_begin();
  input2.halo().exchange_begin();
  input1.halo().exchange_finalize();
  input2.halo().exchange_finalize();

  if (first1.conforms(result) && first2.conforms(result)) {
    std::transform(first1.local(), last1.local(), first2.local(),
                   result.local(), op);
    input1.comm().barrier();
  } else {
    if (input1.comm().rank() == 0) {
      std::transform(first1, last1, first2, result, op);
    }
    result.container().fence();
  }

  return result + (last1 - first1);
}

/// Collective transform on a distributed range: 2 in, 1 out
auto transform(mpi_distributed_contiguous_range auto &&r1,
               mpi_distributed_contiguous_range auto &&r2,
               mpi_distributed_contiguous_iterator auto result, auto op) {
  return lib::transform(r1.begin(), r1.end(), r2.begin(), result, op);
}

//
//
// Transform_reduce
//
//

/// Collective transform_reduce on an iterator/sentinel for a distributed range
template <mpi_distributed_contiguous_iterator I, class T>
T transform_reduce(int root, I first, I last, T init, auto reduction_op,
                   auto transform_op) {
  // Each rank reduces its local segment
  auto val = std::transform_reduce(first.local(), last.local(), 0, reduction_op,
                                   transform_op);

  // Gather segment values on root and reduce for final value
  std::vector<T> vals;
  const communicator &comm = first.container().comm();
  comm.gather(val, vals, root);
  if (comm.rank() == root) {
    return std::reduce(vals.begin(), vals.end(), init, reduction_op);
  } else {
    return 0;
  }
}

} // namespace lib
