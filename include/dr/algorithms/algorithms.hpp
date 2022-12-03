namespace lib {

//
//
// Reduce
//
//

/// Collective reduction on iterator/sentinel for a distributed range
template <typename I, typename S, typename T, typename BinaryOp>
T reduce(int root, I input_iterator, S sentinel, T init, BinaryOp &&binary_op) {
  auto &input = input_iterator.object();
  auto &comm = input.comm();

  if (input.congruent(input_iterator, sentinel)) {

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

//
//
// Transform
//
//

/// Collective transform on an iterator/sentinel for a distributed
/// range: 1 in, 1 out
template <typename DistObj, typename UnaryOp>
auto transform(index_iterator<DistObj> input_iterator,
               index_iterator<DistObj> sentinel,
               index_iterator<DistObj> output_iterator, UnaryOp op) {
  auto &input = input_iterator.object();
  auto &output = output_iterator.object();

  if (input.conforms(output) && input.congruent(input_iterator, sentinel) &&
      output.congruent(output_iterator)) {

    // if input and output conform and this is whole vector, then just
    // do a segment-wise transform
    rng::transform(input.local(), output.local().begin(), op);
  } else {
    if (input.comm().rank() == 0) {
      // This is slow, but will always work. Some faster
      // specializations are possible if needed.
      std::transform(input_iterator, sentinel, output_iterator, op);
    }
  }
  return output.end();
}

/// Collective transform on a distributed range: 1 in, 1 out
template <lib::distributed_contiguous_range R, typename OutputIterator,
          typename UnaryOp>
auto transform(R &&input_range, OutputIterator output_iterator, UnaryOp op) {
  return transform(input_range.begin(), input_range.end(), output_iterator, op);
}

/// Collective transform on an iterator/sentinel for a distributed
/// range: 2 in, 1 out
template <typename DistObj, typename BinaryOp>
auto transform(index_iterator<DistObj> first1, index_iterator<DistObj> last1,
               index_iterator<DistObj> first2, index_iterator<DistObj> d_first,
               BinaryOp op) {
  auto &input1 = first1.object();
  auto &input2 = first2.object();
  auto &output = d_first.object();

  if (input1.conforms(output) && input1.conforms(input2) &&
      input1.congruent(first1, last1) && output.congruent(d_first)) {

    // if input and output conform and this is whole vector, then just
    // do a segment-wise transform
    rng::transform(input1.local(), input2.local(), output.local().begin(), op);
  } else {
    if (input1.comm().rank() == 0) {
      // This is slow, but will always work. Some faster
      // specializations are possible if needed.
      std::transform(first1, last1, first2, d_first, op);
    }
  }
  return output.end();
}

/// Collective transform on a distributed range: 2 in, 1 out
template <lib::distributed_contiguous_range R1,
          lib::distributed_contiguous_range R2, typename O, typename BinaryOp>
auto transform(R1 &&r1, R2 &&r2, O output, BinaryOp op) {
  return transform(r1.begin(), r1.end(), r2.begin(), output, op);
}

//
//
// Transform_reduce
//
//

/// Collective transform_reduce on an iterator/sentinel for a distributed range
template <typename I, class T, typename BinaryReductionOp,
          typename UnaryTransformOp>
T transform_reduce(int root, I input_iterator, I sentinel, T init,
                   BinaryReductionOp reduction_op,
                   UnaryTransformOp transform_op) {
  auto &input = input_iterator.object();
  auto &comm = input.comm();

  if (input.congruent(input_iterator, sentinel)) {

    // Each rank reduces its local segment
    T val = std::transform_reduce(input.local().begin(), input.local().end(), 0,
                                  reduction_op, transform_op);

    // Gather segment values on root and reduce for final value
    std::vector<T> vals;
    comm.gather(val, vals, root);
    if (comm.rank() == root) {
      return std::reduce(vals.begin(), vals.end(), init, reduction_op);
    } else {
      return 0;
    }

  } else {

    // Fall back to std::reduce performing elementwise operations
    if (comm.rank() == root) {
      return std::transform_reduce(input_iterator, sentinel, init, reduction_op,
                                   transform_op);
    } else {
      return 0;
    }
  }
}

} // namespace lib
