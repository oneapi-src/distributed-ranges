// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

namespace mhp {

//
//
// Reduce
//
//

//
// Ranges
//

/// Collective reduction on a distributed range
template <typename ExecutionPolicy, typename T, lib::distributed_range DR>
auto reduce(ExecutionPolicy &&policy, DR &&dr, T init, auto &&binary_op,
            std::optional<std::size_t> root = std::optional<std::size_t>()) {
  T result = 0;
  auto comm = default_comm();

  if (aligned(dr)) {
    lib::drlog.debug("Parallel reduce\n");

    auto reduce = [=](auto &&r) {
      if constexpr (std::is_same_v<std::remove_cvref_t<ExecutionPolicy>,
                                   device_policy>) {
        lib::drlog.debug("  with DPL\n");
        return std::reduce(policy.dpl_policy, &*rng::begin(r), &*rng::end(r),
                           T(0), binary_op);
      } else {
        lib::drlog.debug("  with CPU\n");
        return std::reduce(std::execution::par_unseq, rng::begin(r),
                           rng::end(r), T(0), binary_op);
      }
    };
    auto locals = rng::views::transform(local_segments(dr), reduce);
    auto local = std::reduce(std::execution::par_unseq, rng::begin(locals),
                             rng::end(locals), T(0), binary_op);

    // Reduce locally, gather, reduce globally
    std::vector<T> all(comm.size()); // dr-style ignore
    // If root is provided, final reduce on root
    if (root) {
      comm.gather(local, all, *root);
      if (*root == comm.rank()) {
        result = std::reduce(std::execution::par_unseq, rng::begin(all),
                             rng::end(all), init, binary_op);
      }
    } else {
      comm.all_gather(local, all);
      result = std::reduce(std::execution::par_unseq, rng::begin(all),
                           rng::end(all), init, binary_op);
    }
    lib::drlog.debug("  locals: {}\n"
                     "  local: {}\n"
                     "  all: {}\n"
                     "  result: {}\n",
                     locals, local, all, result);
  } else {
    lib::drlog.debug("Serial reduce\n");
    result = std::reduce(std::execution::par_unseq, rng::begin(dr),
                         rng::begin(dr), init, binary_op);
    barrier();
  }
  return result;
}

/// Collective reduction on a distributed range
template <typename ExecutionPolicy, lib::distributed_range DR>
auto reduce(ExecutionPolicy &&policy, DR &dr, auto init) {
  return reduce(std::forward<ExecutionPolicy>(policy), std::forward<DR>(dr),
                init, std::plus<>{});
}

/// Collective reduction on a distributed range
template <typename ExecutionPolicy, lib::distributed_range DR>
auto reduce(ExecutionPolicy &&policy, DR &dr) {
  return reduce(std::forward<ExecutionPolicy>(policy), std::forward<DR>(dr),
                typename DR::value_type{}, std::plus<>{});
}

//
// Iterators
//

/// Collective reduction on a distributed range
template <typename ExecutionPolicy, lib::distributed_iterator DI,
          typename BinaryOp>
auto reduce(ExecutionPolicy &&policy, DI begin, DI end,
            auto init = typename DI::value_type{},
            BinaryOp &&binary_op = std::plus<>(),
            std::optional<std::size_t> root = std::optional<std::size_t>()) {
  return reduce(std::forward<ExecutionPolicy>(policy),
                rng::subrange(begin, end), init,
                std::forward<BinaryOp>(binary_op), root);
}

/// Collective reduction on a distributed range
template <typename ExecutionPolicy, lib::distributed_iterator DI>
auto reduce(ExecutionPolicy &&policy, DI begin, DI end,
            auto init = typename DI::value_type{}) {
  return reduce(std::forward<ExecutionPolicy>(policy),
                rng::subrange(begin, end), init, std::plus<>{});
}

/// Collective reduction on a distributed range
template <typename ExecutionPolicy, lib::distributed_iterator DI>
auto reduce(ExecutionPolicy &&policy, DI begin, DI end) {
  return reduce(std::forward<ExecutionPolicy>(policy),
                rng::subrange(begin, end), typename DI::value_type{},
                std::plus<>{});
}

} // namespace mhp
