#include "cpu-tests.hpp"

TEST(CpuMpiTests, DistributedVectorRequirements) {
  using DV = lib::distributed_vector<int>;
  assert_distributed_range<DV>();
}

TEST(CpuMpiTests, DistributedVectorConstructors) {
  lib::distributed_vector<int> a1(10);

  lib::distributed_vector<int> a2(10);
}

TEST(CpuMpiTests, DistributedVectorQuery) {
  const int n = 10;
  lib::distributed_vector<int> a(n);

  EXPECT_EQ(a.size(), n);
}

TEST(CpuMpiTests, DistributedVectorGatherScatter) {
  const std::size_t n = 10;
  const int root = 0;
  lib::distributed_vector<int> dv(n);

  std::vector<int> src(n), dst(n);
  std::iota(src.data(), src.data() + src.size(), 1);

  dv.scatter(src, root);
  dv.gather(dst, root);

  expect_eq(src, dst, root);
}

TEST(CpuMpiTests, distributed_vector_index) {
  const std::size_t n = 10;
  lib::distributed_vector<int> dv(n);

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

  lib::distributed_vector<int> dv2(n);

  dv2[3] = 1000;
  dv2[3] = dv[3];
  EXPECT_EQ(dv2[3], dv[3]);
}

TEST(CpuMpiTests, DistributedVectorCollectiveCopy) {
  const std::size_t n = 10;
  const int root = 0;
  lib::distributed_vector<int> dv(n);

  std::vector<int> src(n), dst(n);
  std::iota(src.data(), src.data() + src.size(), 1);

  lib::collective::copy(root, src, dv);
  lib::collective::copy(root, dv, dst);

  expect_eq(src, dst, root);
}

TEST(CpuMpiTests, DistributedVectorAlgorithms) {
  const std::size_t n = 10;
  const int root = 0;
  lib::distributed_vector<int> dv(n);

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
}

using DVX = lib::distributed_vector<int>;

int a;

// Operations on a const distributed_vector
void common_operations(auto &dv) {
  a = dv[1];
  EXPECT_EQ(dv[1], 101);
  EXPECT_EQ(*(&(dv[1])), 101);

  auto p = &dv[1];
  EXPECT_EQ(*(p + 1), 102);
}

TEST(CpuMpiTests, DistributedVectorXreference) {
  std::size_t n = 10;
  DVX dv(n);
  rng::iota(dv, 100);
  const DVX &cdv = dv;
  if (comm_rank == 0) {
    common_operations(cdv);
    common_operations(dv);
  }

  dv[2] = 2;
  EXPECT_EQ(dv[2], 2);

  static_assert(std::random_access_iterator<DVX::iterator>);
  static_assert(std::random_access_iterator<DVX::const_iterator>);
}
