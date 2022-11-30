namespace lib {

namespace collective {

/// Reduction
template <typename T, typename DistributedVector, typename BinaryOp>
T reduce(int root, DistributedVector &&dv, T init, BinaryOp &&binary_op) {

  auto &comm = dv.comm();

  T val = std::reduce(dv.local().begin(), dv.local().end(), 0, binary_op);
  drlog.debug("local reduce: {}\n", val);
  std::vector<T> vals(comm.size());
  comm.gather(&val, vals.data(), sizeof(T), root);
  if (comm.rank() != root)
    return 0;

  return std::reduce(vals.begin(), vals.end(), init, binary_op);
}

} // namespace collective

} // namespace lib
