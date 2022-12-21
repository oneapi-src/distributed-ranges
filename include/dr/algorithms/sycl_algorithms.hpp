// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

namespace lib {

//
//
// Reduce
//
//

///
/// Reduce SYCL distributed vector
template <sycl_mpi_distributed_contiguous_iterator SDI, typename T>
T reduce(int root, SDI first, SDI last, T init, auto &&binary_op) {
  auto &container = first.container();
  auto &comm = container.comm();

  // Each rank reduces its local segment
  auto val = std::reduce(container.allocator().policy(), first.local(),
                         last.local(), 0, binary_op);
  drlog.debug("dpl local reduce: {}\n", val);

  // Gather segment values on root and reduce for final value
  std::vector<T> vals;
  comm.gather(val, vals, root);
  if (comm.rank() == root) {
    return std::reduce(vals.begin(), vals.end(), init, binary_op);
  } else {
    return 0;
  }
}

//
//
// Transform
//
//

/// Collective transform on an iterator/sentinel for a distributed
/// SYCL vector. range: 1 in, 1 out
template <sycl_mpi_distributed_contiguous_iterator SDI>
auto transform(SDI first, SDI last,
               sycl_mpi_distributed_contiguous_iterator auto result, auto op) {
  auto &input = first.container();

  input.halo().exchange_begin();
  input.halo().exchange_finalize();
  if (first.conforms(result)) {
    // DPL does copies data if you don't pass a pointer??
    std::transform(input.allocator().policy(), &*first.local(), &*last.local(),
                   &*result.local(), op);
    drlog.debug("dpl local transform\n");
  } else {
    if (input.comm().rank() == 0) {
      std::transform(first, last, result, op);
    }
  }
  return result + (last - first);
}

} // namespace lib
