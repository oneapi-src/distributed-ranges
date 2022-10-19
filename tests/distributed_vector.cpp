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

void expect_range_eq(int root, auto &r1, auto &r2) {
  if (comm_rank == root) {
    for (size_t i = 0; i < r1.size(); i++) {
      EXPECT_EQ(r1[i], r2[i]);
    }
  }
}

TEST(CpuMpiTest, DistributedVectorGatherScatter) {
  const std::size_t n = 10;
  const int root = 0;
  auto dist = lib::block_cyclic(lib::partition_method::div, comm);
  lib::distributed_vector<int, lib::block_cyclic> dv(n, dist);

  std::vector<int> src(n), dst(n);
  std::iota(src.data(), src.data() + src.size(), 1);

  dv.scatter(src, root);
  dv.gather(dst, root);

  expect_range_eq(root, src, dst);
}

TEST(CpuMpiTest, DistributedVectorCopy) {
  const std::size_t n = 10;
  const int root = 0;
  auto dist = lib::block_cyclic(lib::partition_method::div, comm);
  lib::distributed_vector<int, lib::block_cyclic> dv(n, dist);

  std::vector<int> src(n), dst(n);
  std::iota(src.data(), src.data() + src.size(), 1);

  lib::copy(lib::collective_root_policy{root}, src, dv);
  lib::copy(lib::collective_root_policy{root}, dv, dst);

  expect_range_eq(root, src, dst);
}
