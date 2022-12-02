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

TEST(CpuMpiTests, TransformDistributedVector2) {
  std::size_t n = 10;
  auto op = std::plus<>();

  std::vector<int> v1(n), v2(n), vr(n);
  lib::distributed_vector<int> dv1(n), dv2(n), dvr0(n), dvr1(n);

  rng::iota(v1, 100);
  rng::iota(v2, 200);
  if (comm_rank == 0) {
    rng::copy(v1, dv1.begin());
    rng::copy(v2, dv2.begin());
  }
  dv1.fence();
  dv2.fence();

  if (comm_rank == 0) {
    rng::transform(v1, v2, vr.begin(), op);
    rng::transform(dv1, dv2, dvr0.begin(), op);
    expect_eq(dvr0, vr);
  }

  lib::transform(dv1, dv2, dvr1.begin(), op);
  if (comm_rank == 0) {
    expect_eq(dvr1, vr);
  }
}
