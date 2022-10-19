#include "cpu-mpi-tests.hpp"

TEST(CpuMpiTests, DistributedVectorRequirements) {
  using DV = lib::distributed_vector<int>;

  static_assert(rng::range<DV>);
  static_assert(lib::distributed_contiguous_range<DV>);

  // DV dv(10, lib::block_cyclic(lib::partition::div));
}

TEST(CpuMpiTest, DistributedVectorConstructors) {
  auto dist = lib::block_cyclic(lib::partition_method::div, comm);
  lib::distributed_vector<int, lib::block_cyclic> a(10, dist);
}

TEST(CpuMpiTest, DistributedVectorGatherScatter) {
  const std::size_t n = 10;
  const int root = 0;
  auto dist = lib::block_cyclic(lib::partition_method::div, comm);
  lib::distributed_vector<int, lib::block_cyclic> a(n, dist);

  std::vector<int> src(n), dst(n);
  std::iota(src.data(), src.data() + src.size(), 1);

  a.scatter(src, root);

  a.gather(dst, root);
  if (comm_rank == root) {
    for (size_t i = 0; i < src.size(); i++) {
      EXPECT_EQ(src[i], dst[i]);
    }
  }
}
