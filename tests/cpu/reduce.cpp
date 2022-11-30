#include "cpu-tests.hpp"

TEST(CpuMpiTests, ReduceDistributedVector) {
  std::size_t n = 10;

  std::vector<int> v(n);
  rng::iota(v, 100);

  lib::distributed_vector<int> dv(n);
  if (comm_rank == 0) {
    rng::copy(v, dv.begin());
  }
  dv.fence();

  auto dval = lib::collective::reduce(0, dv, 10000, std::plus<>());
  if (comm_rank == 0) {
    auto lval = std::reduce(v.begin(), v.end(), 10000, std::plus<>());
    EXPECT_EQ(dval, lval);
  }
}
