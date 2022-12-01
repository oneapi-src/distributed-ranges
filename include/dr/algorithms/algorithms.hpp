namespace lib {

/// Collective reduction on interator/sentinel for a distributed range
template <typename I, typename S, typename T, typename BinaryOp>
T reduce(int root, I input_iterator, S sentinel, T init, BinaryOp &&binary_op) {
  auto &input = input_iterator.object();
  auto &comm = input.comm();

  if (input_iterator == input.begin() && sentinel == input.end()) {

    // Each rank reduces its local segment
    T val =
        std::reduce(input.local().begin(), input.local().end(), 0, binary_op);
    drlog.debug("local reduce: {}\n", val);

    // Gather segment values on root and reduce for final value
    std::vector<T> vals;
    comm.gather(val, vals, root);
    if (comm.rank() == root) {
      return std::reduce(vals.begin(), vals.end(), init, binary_op);
    } else {
      return 0;
    }

  } else {

    // Fall back to std::reduce performing elementwise operations
    if (comm.rank() == root) {
      return std::reduce(input_iterator, sentinel, init, binary_op);
    } else {
      return 0;
    }
  }
}

/// Collective reduction on a distributed range
template <lib::distributed_contiguous_range R, typename T, typename BinaryOp>
T reduce(int root, R &&r, T init, BinaryOp &&binary_op) {
  return reduce(root, r.begin(), r.end(), init, binary_op);
}

/// Collective transform on an iterator/sentinel for a distributed range
template <typename I, typename S, typename O, typename UnaryOp>
auto transform(I input_iterator, S sentinel, O output_iterator, UnaryOp op) {
  auto &input = input_iterator.object();
  auto &output = output_iterator.object();

  if (input.conforms(output) && input_iterator == input.begin() &&
      sentinel == input.end() && output_iterator == output.begin) {

    // if input and output conform and this is whole vector, then just
    // do a segment-wise transform
    rng::transform(input.local(), output.local().begin(), op);
  } else {
    if (input.comm().rank() == 0) {
      // This is slow, but will always work. Some faster
      // specializations are possible if needed.
      std::transform(input_iterator, sentinel, output_iterator, op);
    }
    return output.end();
  }
}

/// Collective transform on a distributed range
template <lib::distributed_contiguous_range R, typename OutputIterator,
          typename UnaryOp>
auto transform(R &&input_range, OutputIterator output_iterator, UnaryOp op) {
  return transform(input_range.begin(), input_range.end(), output_iterator, op);
}

} // namespace lib
