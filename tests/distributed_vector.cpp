#include "cpu-mpi-tests.hpp"

TEST(CpuMpiTests, DistributedVectorRequirements) {
  using DV = lib::distributed_vector<int>;

  static_assert(rng::range<DV>);
  static_assert(lib::distributed_contiguous_range<DV>);

  // DV dv(10, lib::block_cyclic(lib::partition::div));
}
