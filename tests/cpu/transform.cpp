#include "cpu-tests.hpp"

TEST(CpuMpiTests, TransformDistributedVector) {
  std::size_t n = 10;
  auto square = [](auto n) { return n * n; };

  std::vector<int> vi(n), vr(n);
  lib::distributed_vector<int> dvi(n), dvr(n);

  rng::iota(vi, 100);
  if (comm_rank == 0) {
    rng::copy(vi, dvi.begin());
  }
  dvi.fence();

  if (comm_rank == 0) {
    rng::transform(vi, vr.begin(), square);
    rng::transform(dvi, dvr.begin(), square);
    expect_eq(dvr, vr);
  }

  lib::transform(dvi, dvr.begin(), square);
  if (comm_rank == 0) {
    expect_eq(dvr, vr);
  }
}
