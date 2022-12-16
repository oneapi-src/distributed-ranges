#include "cpu-tests.hpp"

void check_fill(std::size_t n, std::size_t b, std::size_t e) {
  int val = 33;

  lib::distributed_vector<int> dv1(n), dv2(n), dv3(n);
  lib::fill(dv1.begin() + b, dv1.begin() + e, val);
  lib::fill(rng::subrange(dv3.begin() + b, dv3.begin() + e), val);

  if (comm_rank == 0) {
    std::fill(dv2.begin() + b, dv2.begin() + e, val);

    std::vector<int> v(n);
    std::fill(v.begin() + b, v.begin() + e, val);

    EXPECT_TRUE(equal(dv1, v));
    EXPECT_TRUE(equal(dv2, v));
    EXPECT_TRUE(equal(dv3, v));
  }
}

TEST(CpuMpiTests, FillDistributedVector) {
  std::size_t n = 10;

  check_fill(n, 0, n);
  check_fill(n, n / 2 - 1, n / 2 + 1);
}
