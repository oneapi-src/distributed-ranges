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
//
// Reduce
//
//

/// Collective reduction on a distributed range
template <typename R, typename T, typename BinaryOp>
T reduce(int root, R &&r, T init, BinaryOp &&binary_op) {
  auto lr = r | local_span();
  auto val = std::reduce(lr.begin(), lr.end(), 0, binary_op);
  drlog.debug("local reduce: {}\n", val);

  // Gather segment values on root and reduce for final value
  std::vector<T> vals;
  auto &comm = r.begin().container().comm();
  comm.gather(val, vals, root);
  if (comm.rank() == root) {
    auto gval = std::reduce(vals.begin(), vals.end(), init, binary_op);
    drlog.debug("global reduce: {}\n", gval);
    return gval;
  } else {
    return 0;
  }
}

/// Collective reduction on iterator/sentinel for a distributed range
template <distributed_contiguous_iterator I, typename T, typename BinaryOp>
T reduce(int root, I first, I last, T init, BinaryOp &&binary_op) {
  return lib::reduce(root, rng::subrange(first, last), init, binary_op);
}

//
//
// Copy
//
//

/// Collective copy from distributed range to distributed iterator
template <typename R, distributed_contiguous_iterator O>
void copy(R &&r, O result) {
  if (r.begin().conforms(result)) {
    rng::copy(r | local_span(), result.local());
  } else {
    rng::copy(r, result);
  }
}

/// Collective copy from distributed iterator to distributed iterator
template <distributed_contiguous_iterator I, distributed_contiguous_iterator O>
void copy(I first, I last, O result) {
  lib::copy(rng::subrange(first, last), result);
}

//
//
// Transform
//
//

/// Collective transform on a distributed range: 1 in, 1 out
template <typename R, distributed_contiguous_iterator O, typename UnaryOp>
auto transform(R &&r, O result, UnaryOp op) {
  auto &input = r.begin().container();

  input.halo().exchange_begin();
  input.halo().exchange_finalize();
  if (result.conforms(r.begin())) {
    rng::transform(r | local_span(), result.local(), op);
  } else {
    if (input.comm().rank() == 0) {
      rng::transform(r, result, op);
    }
  }
  return result + (r.end() - r.begin());
}

/// Collective transform on an iterator/sentinel for a distributed
/// range: 1 in, 1 out
template <distributed_contiguous_iterator I, distributed_contiguous_iterator O,
          typename UnaryOp>
auto transform(I first, I last, O result, UnaryOp op) {
  return lib::transform(rng::subrange(first, last), result, op);
}

/// Collective transform on a distributed range: 2 in, 1 out
// template <distributed_contiguous_range R1, distributed_contiguous_range R2,
template <typename R1, typename R2, distributed_contiguous_iterator O,
          typename BinaryOp>
auto transform(R1 &&r1, R2 &&r2, O result, BinaryOp op) {
  auto &input1 = r1.begin().container();
  auto &input2 = r2.begin().container();

  input1.halo().exchange_begin();
  input2.halo().exchange_begin();
  input1.halo().exchange_finalize();
  input2.halo().exchange_finalize();

  if (result.conforms(r1.begin()) && result.conforms(r2.begin())) {
    rng::transform(r1 | local_span(), r2 | local_span(), result.local(), op);
  } else {
    if (input1.comm().rank() == 0) {
      rng::transform(r1, r2, result, op);
    }
  }

  return result + (r1.end() - r1.begin());
}

/// Collective transform on an iterator/sentinel for a distributed
/// range: 2 in, 1 out
template <distributed_contiguous_iterator I1,
          distributed_contiguous_iterator I2, distributed_contiguous_iterator O,
          typename BinaryOp>
auto transform(I1 first1, I1 last1, I2 first2, O result, BinaryOp op) {
  return lib::transform(rng::subrange(first1, last1),
                        rng::subrange(first2, I2{}), result, op);
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
