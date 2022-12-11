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

//
//
// Transform
//
//

/// Exclude from doxygen
/// \cond
/// Collective transform on an iterator/sentinel for a distributed
/// range: 1 in, 1 out
template <sycl_distributed_contiguous_iterator InputIt,
          sycl_distributed_contiguous_iterator OutputIt, typename UnaryOp>
auto transform(InputIt first, InputIt last, OutputIt d_first, UnaryOp op) {
  auto &input = first.object();
  auto &output = d_first.object();
  auto &comm = input.comm();

  input.halo().exchange_begin();
  input.halo().exchange_finalize();
  if (input.conforms(output) && first.index_ == d_first.index_) {
    auto [begin_offset, end_offset] =
        input.select_local(first, last, comm.rank());
    // if input and output conform and this is whole vector, then just
    // do a segment-wise transform
    std::transform(input.allocator().policy(),
                   input.local().data() + begin_offset,
                   input.local().data() + end_offset,
                   output.local().data() + begin_offset, op);
    drlog.debug("dpl local transform\n");

  } else {
    if (input.comm().rank() == 0) {
      // This is slow, but will always work. Some faster
      // specializations are possible if needed.
      std::transform(first, last, d_first, op);
    }
  }
  return output.end();
}
/// \endcond

} // namespace lib
