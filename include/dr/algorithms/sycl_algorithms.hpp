namespace lib {

//
//
// Reduce
//
//

/// Exclude this from doxygen because signature is identical to
/// non-sycl version
///
/// \cond
template <sycl_distributed_contiguous_iterator I, typename T, typename BinaryOp>
T reduce(int root, I first, I last, T init, BinaryOp &&binary_op) {
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
/// \endcond

//
//
// Transform
//
//

/// Exclude from doxygen
/// \cond
/// Collective transform on an iterator/sentinel for a distributed
/// range: 1 in, 1 out
template <sycl_distributed_contiguous_iterator I,
          sycl_distributed_contiguous_iterator O, typename UnaryOp>
auto transform(I first, I last, O result, UnaryOp op) {
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
/// \endcond

} // namespace lib
