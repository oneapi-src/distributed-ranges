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

TEST(CpuMpiTests, Stencil) {
  std::size_t n = 10;
  auto op = [](auto &&v) {
    auto p = &v;
    return p[-1] + p[0] + p[+1];
  };

  std::vector<int> v_in(n), v_out(n);
  lib::stencil<1> s(1);
  lib::distributed_vector<int> dv_in(s, n), dv_out1(s, n), dv_out2(s, n);

  if (comm_rank == 0) {
    rng::iota(v_in, 100);
    rng::copy(v_in, dv_in.begin());
  }
  dv_in.fence();

  if (comm_rank == 0) {
    rng::transform(v_in.begin() + 1, v_in.end() - 1, v_out.begin() + 1, op);
    rng::transform(dv_in.begin() + 1, dv_in.end() - 1, dv_out1.begin() + 1, op);
    expect_eq(dv_out1, v_out);
  }

  fmt::print("Initial:   {}\n", v_in);
  fmt::print("Reference: {}\n", v_out);
  fmt::print("Test:      {}\n", dv_out1);
  // lib::transform(dv_in.begin() + 1, dv_in.end() - 1, dv_out2.begin() + 1,
  // op); if (comm_rank == 0) { expect_eq(dv_out2, v_out);
  // }
}
