#include "cpu-tests.hpp"

TEST(CpuMpiTests, DistributedVectorRequirements) {
  using DV = lib::distributed_vector<int>;

  static_assert(std::random_access_iterator<lib::index_iterator<DV>>);
  static_assert(rng::range<DV>);
  static_assert(lib::distributed_contiguous_range<DV>);

  // DV dv(10, lib::block_cyclic(lib::partition::div));
}

TEST(CpuMpiTests, DistributedVectorConstructors) {
  auto dist = lib::block_cyclic(lib::partition_method::div, comm);
  lib::distributed_vector<int, lib::block_cyclic> a(10, dist);
}

TEST(CpuMpiTests, DistributedVectorQuery) {
  const int n = 10;
  auto dist = lib::block_cyclic(lib::partition_method::div, comm);
  lib::distributed_vector<int, lib::block_cyclic> a(n, dist);

  EXPECT_EQ(a.size(), n);
}

TEST(CpuMpiTests, DistributedVectorGatherScatter) {
  const std::size_t n = 10;
  const int root = 0;
  auto dist = lib::block_cyclic(lib::partition_method::div, comm);
  lib::distributed_vector<int, lib::block_cyclic> dv(n, dist);

  std::vector<int> src(n), dst(n);
  std::iota(src.data(), src.data() + src.size(), 1);

  dv.scatter(src, root);
  dv.gather(dst, root);

  expect_eq(src, dst, root);
}

TEST(CpuMpiTests, DistributedVectorIndex) {
  const std::size_t n = 10;
  // const int root = 1;
  auto dist = lib::block_cyclic(lib::partition_method::div, comm);
  lib::distributed_vector<int, lib::block_cyclic> dv(n, dist);
  dv.fence();

  if (comm_rank == 0) {
    for (size_t i = 0; i < n; i++) {
      dv[i] = i + 10;
    }
  }
  dv.fence();

  if (comm_rank == 0) {
    for (size_t i = 0; i < n; i++) {
      EXPECT_EQ(dv[i], i + 10);
    }
  }

  dv.fence();
  lib::drlog.debug("Done\n");
}

TEST(CpuMpiTests, DistributedVectorCollectiveCopy) {
  const std::size_t n = 10;
  const int root = 0;
  auto dist = lib::block_cyclic(lib::partition_method::div, comm);
  lib::distributed_vector<int, lib::block_cyclic> dv(n, dist);

  std::vector<int> src(n), dst(n);
  std::iota(src.data(), src.data() + src.size(), 1);

  lib::collective::copy(root, src, dv);
  lib::collective::copy(root, dv, dst);

  expect_eq(src, dst, root);
}

TEST(CpuMpiTests, DistributedVectorAlgorithms) {
  const std::size_t n = 10;
  const int root = 0;
  auto dist = lib::block_cyclic(lib::partition_method::div, comm);
  lib::distributed_vector<int, lib::block_cyclic> dv(n, dist);
  dv.fence();

  if (comm_rank == root) {
    std::vector<int> ref(n);
    std::iota(ref.begin(), ref.end(), 1);

    std::iota(dv.begin(), dv.end(), 1);

    expect_eq(dv, ref);

    std::iota(ref.begin(), ref.end(), 11);
    std::copy(ref.begin(), ref.end(), dv.begin());
    expect_eq(dv, ref);

    std::iota(ref.begin(), ref.end(), 21);
    rng::copy(ref, dv.begin());
    expect_eq(dv, ref);

    std::iota(dv.begin(), dv.end(), 31);
    rng::copy(dv, ref.begin());
    expect_eq(dv, ref);
  }

  dv.fence();
}
