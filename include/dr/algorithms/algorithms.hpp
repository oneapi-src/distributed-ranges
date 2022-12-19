namespace lib {

//
//
// Fill
//
//

/// Collective fill on iterator/sentinel for a distributed range
template <mpi_distributed_contiguous_iterator DI>
void fill(DI first, DI last, auto value) {
  rng::fill(first.local(), last.local(), value);
}

/// Collective fill on distributed range
void fill(mpi_distributed_contiguous_range auto &&r, auto value) {
  lib::fill(r.begin(), r.end(), value);
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
  auto &comm = first.container().comm();
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
  } else {
    if (first.container().comm().rank() == 0) {
      std::copy(first, last, result);
    }
  }
}

/// Collective copy from distributed range to distributed iterator
void copy(mpi_distributed_contiguous_range auto &&r,
          mpi_distributed_contiguous_iterator auto result) {
  lib::copy(r.begin(), r.end(), result);
}

auto scatter_data(mpi_distributed_contiguous_iterator auto first,
                  std::size_t size, std::vector<int> &counts,
                  std::vector<int> &offsets) {
  auto comm = first.container().comm();
  counts.resize(comm.size());
  offsets.resize(comm.size());

  auto last = first + size;
  std::size_t offset = 0;
  std::size_t element_size = sizeof(std::iter_value_t<decltype(first)>);
  for (int rank = 0; rank < comm.size(); rank++) {
    auto b = first.remote_offset(rank);
    auto e = last.remote_offset(rank);
    auto size = (e - b) * element_size;
    counts[rank] = size;
    offsets[rank] = offset;
    offset += size;
  }

  drlog.debug("scatter data:\n  counts: {}\n  offsets: {}\n", counts, offsets);
}

/// Collective copy from local begin/end to distributed
template <typename I>
void copy(int root, I first, I last,
          mpi_distributed_contiguous_iterator auto result) {
  auto &comm = result.container().comm();
  std::vector<int> counts(comm.size()), offsets(comm.size());

  scatter_data(result, last - first, counts, offsets);
  comm.scatterv(&*first, counts.data(), offsets.data(), &*result.local(), root);
}

/// Collective copy from local begin/end to distributed
template <typename I>
void copy(int root, I first, std::size_t size,
          mpi_distributed_contiguous_iterator auto result) {
  lib::copy(root, first, first + size, result);
}

/// Collective copy from local range to distributed iterator
void copy(int root, rng::contiguous_range auto &&r,
          mpi_distributed_contiguous_iterator auto result) {
  lib::copy(root, r.begin(), r.end(), result);
}

/// Collective copy from distributed begin/end to local iterator
template <mpi_distributed_contiguous_iterator DI>
void copy(int root, DI first, DI last, std::contiguous_iterator auto result) {
  auto &comm = first.container().comm();
  std::vector<int> counts(comm.size()), offsets(comm.size());

  scatter_data(first, last - first, counts, offsets);
  comm.gatherv(&*first.local(), counts.data(), offsets.data(), &*result, root);
}

/// Collective copy from distributed begin/end to local iterator
template <mpi_distributed_contiguous_iterator DI>
void copy(int root, DI first, std::size_t size,
          std::contiguous_iterator auto result) {
  lib::copy(root, first, first + size, result);
}

/// Collective copy from distributed range to local iterator
void copy(int root, mpi_distributed_contiguous_range auto &&r, auto result) {
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
  } else {
    if (input1.comm().rank() == 0) {
      std::transform(first1, last1, first2, result, op);
    }
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
  auto &comm = first.container().comm();
  comm.gather(val, vals, root);
  if (comm.rank() == root) {
    return std::reduce(vals.begin(), vals.end(), init, reduction_op);
  } else {
    return 0;
  }
}

} // namespace lib
