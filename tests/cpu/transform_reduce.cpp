#include "cpu-tests.hpp"

TEST(CpuMpiTests, TransformReduceDistributedVector) {
  std::size_t n = 10;
  int init = 10000;
  int lval, dval;
  auto reduce_op = std::plus<>();
  auto transform_op = [](auto v) { return v * v; };

  std::vector<int> v(n);
  lib::distributed_vector<int> dv(n);

  rng::iota(v, 100);
  if (comm_rank == 0) {
    rng::copy(v, dv.begin());
  }
  dv.fence();

  static_assert(
      std::random_access_iterator<lib::distributed_vector<int>::iterator>);
  if (comm_rank == 0) {
    lval = std::transform_reduce(v.begin(), v.end(), init, reduce_op,
                                 transform_op);
    dval = std::transform_reduce(dv.begin(), dv.end(), init, reduce_op,
                                 transform_op);
    EXPECT_EQ(dval, lval);
  }

  dval = lib::transform_reduce(0, dv.begin(), dv.end(), init, reduce_op,
                               transform_op);
  if (comm_rank == 0) {
    EXPECT_EQ(dval, lval);
  }
}
