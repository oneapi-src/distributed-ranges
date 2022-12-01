#include "cpu-tests.hpp"

TEST(CpuMpiTests, ReduceDistributedVector) {
  std::size_t n = 10;
  auto op = std::plus<>();
  int init = 10000;
  int lval, dval;

  std::vector<int> v(n);
  lib::distributed_vector<int> dv(n);

  rng::iota(v, 100);
  if (comm_rank == 0) {
    rng::copy(v, dv.begin());
  }
  dv.fence();

  if (comm_rank == 0) {
    lval = std::reduce(v.begin(), v.end(), init, op);
    dval = std::reduce(dv.begin(), dv.end(), init, op);
    EXPECT_EQ(dval, lval);
  }

  dval = lib::reduce(0, dv, init, op);
  if (comm_rank == 0) {
    EXPECT_EQ(dval, lval);
  }
}
