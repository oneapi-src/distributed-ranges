namespace lib {

//
//
// Fill
//
//

/// Collective fill on distributed range
template <typename R, typename T> void fill(R &&r, T value) {
  rng::fill(r | local_span(), value);
}

/// Collective fill on iterator/sentinel for a distributed range
template <distributed_contiguous_iterator I, typename T>
void fill(I first, I last, T value) {
  lib::fill(rng::subrange(first, last), value);
}

//
// Reduce
//
//

/// Collective reduction on iterator/sentinel for a distributed range
template <distributed_contiguous_iterator I, typename T, typename BinaryOp>
T reduce(int root, I first, I last, T init, BinaryOp &&binary_op) {
  auto &container = first.object();
  auto &comm = container.comm();
  auto [begin_offset, end_offset] =
      container.select_local(first, last, container.comm().rank());
  auto base = container.local().begin();

  // Each rank reduces its local segment
  T val = std::reduce(base + begin_offset, base + end_offset, 0, binary_op);
  drlog.debug("local reduce: {}\n", val);

  // Gather segment values on root and reduce for final value
  std::vector<T> vals;
  comm.gather(val, vals, root);
  if (comm.rank() == root) {
    return std::reduce(vals.begin(), vals.end(), init, binary_op);
  } else {
    return 0;
  }
}

/// Collective reduction on a distributed range
template <distributed_contiguous_range R, typename T, typename BinaryOp>
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
template <distributed_contiguous_iterator InputIt,
          distributed_contiguous_iterator OutputIt, typename UnaryOp>
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
    std::transform(input.local().begin() + begin_offset,
                   input.local().begin() + end_offset,
                   output.local().begin() + begin_offset, op);
  } else {
    if (input.comm().rank() == 0) {
      // This is slow, but will always work. Some faster
      // specializations are possible if needed.
      std::transform(first, last, d_first, op);
    }
  }
  return output.end();
}

/// Collective transform on a distributed range: 1 in, 1 out
template <distributed_contiguous_range R, typename OutputIterator,
          typename UnaryOp>
auto transform(R &&input_range, OutputIterator output_iterator, UnaryOp op) {
  return transform(input_range.begin(), input_range.end(), output_iterator, op);
}

/// Collective transform on an iterator/sentinel for a distributed
/// range: 2 in, 1 out
template <distributed_contiguous_iterator InputIt1,
          distributed_contiguous_iterator InputIt2,
          distributed_contiguous_iterator OutputIt, typename BinaryOp>
auto transform(InputIt1 first1, InputIt1 last1, InputIt2 first2,
               OutputIt d_first, BinaryOp op) {
  auto &input1 = first1.object();
  auto &input2 = first2.object();
  auto &output = d_first.object();
  auto &comm = input1.comm();

  input1.halo().exchange_begin();
  input1.halo().exchange_finalize();
  input2.halo().exchange_begin();
  input2.halo().exchange_finalize();
  if (input1.conforms(output) && input1.conforms(input2) &&
      first1.index_ == first2.index_ && first2.index_ == d_first.index_) {
    auto [begin_offset, end_offset] =
        input1.select_local(first1, last1, comm.rank());
    // if input and output conform and this is whole vector, then just
    // do a segment-wise transform
    std::transform(input1.local().begin() + begin_offset,
                   input1.local().begin() + end_offset,
                   input2.local().begin() + begin_offset,
                   output.local().begin() + begin_offset, op);
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
// template <distributed_contiguous_range R1, distributed_contiguous_range R2,
template <typename R1, typename R2, typename O, typename BinaryOp>
auto transform(R1 &&r1, R2 &&r2, O output, BinaryOp op) {
  return transform(r1.begin(), r1.end(), r2.begin(), output, op);
}

//
//
// Transform_reduce
//
//

/// Collective transform_reduce on an iterator/sentinel for a distributed range
template <distributed_contiguous_iterator I, class T,
          typename BinaryReductionOp, typename UnaryTransformOp>
T transform_reduce(int root, I first, I last, T init,
                   BinaryReductionOp reduction_op,
                   UnaryTransformOp transform_op) {
  auto &input = first.object();
  auto &comm = input.comm();
  auto [begin_offset, end_offset] =
      input.select_local(first, last, comm.rank());
  auto base = input.local().begin();

  // Each rank reduces its local segment
  T val = std::transform_reduce(base + begin_offset, base + end_offset, 0,
                                reduction_op, transform_op);

  // Gather segment values on root and reduce for final value
  std::vector<T> vals;
  comm.gather(val, vals, root);
  if (comm.rank() == root) {
    return std::reduce(vals.begin(), vals.end(), init, reduction_op);
  } else {
    return 0;
  }
}

} // namespace lib
