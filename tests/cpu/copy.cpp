#include "cpu-tests.hpp"

void check_copy(std::size_t n, std::size_t b, std::size_t e) {
  std::vector<int> v_in(n), v(n), v1(n);
  rng::iota(v_in, 100);

  lib::distributed_vector<int> dv_in(n), dv1(n), dv2(n), dv3(n), dv4(n);
  rng::iota(dv_in, 100);
  lib::copy(dv_in.begin() + b, dv_in.begin() + e, dv1.begin() + b);
  lib::copy(rng::subrange(dv_in.begin() + b, dv_in.begin() + e),
            dv2.begin() + b);

  lib::copy(0, v_in.begin() + b, v_in.begin() + e, dv4.begin() + b);
  lib::copy(0, dv_in.begin() + b, dv_in.begin() + e, v1.begin() + b);

  if (comm_rank == 0) {
    std::copy(dv_in.begin() + b, dv_in.begin() + e, dv3.begin() + b);

    std::copy(v_in.begin() + b, v_in.begin() + e, v.begin() + b);

    EXPECT_TRUE(equal(dv1, v));
    EXPECT_TRUE(equal(dv2, v));
    EXPECT_TRUE(equal(dv3, v));
    EXPECT_TRUE(equal(dv4, v));

    EXPECT_TRUE(equal(v1, v));
  }
}

TEST(CpuMpiTests, CopyDistributedVector) {
  std::size_t n = 10;

  check_copy(n, 0, n);
  check_copy(n, n / 2 - 1, n / 2 + 1);
}
