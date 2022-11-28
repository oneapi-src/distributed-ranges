#include "sycl-mpi-tests.hpp"

using DV = lib::distributed_vector<int, lib::shared_allocator<int>>;
TEST(SyclMpiTests, DistributedVector) {
  sycl::queue q;
  lib::shared_allocator<int> q_alloc(q);

  DV dv(q_alloc, 10);
  dv.fence();

  auto p = dv.local().data();
  q.single_task([p]() { *p = 1; }).wait();
  EXPECT_EQ(dv[0], 1);

  dv.fence();
}
