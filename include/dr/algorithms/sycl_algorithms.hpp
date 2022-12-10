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
  auto &container = first.object();
  auto &comm = container.comm();
  auto [begin_offset, end_offset] =
      container.select_local(first, last, container.comm().rank());
  auto base = container.local().begin();

  // Each rank reduces its local segment
  T val = std::reduce(container.allocator().policy(), base + begin_offset,
                      base + end_offset, 0, binary_op);
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

} // namespace lib
