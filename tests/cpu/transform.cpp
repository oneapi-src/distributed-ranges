#include "cpu-tests.hpp"

void check_transform(std::string title, std::size_t n, std::size_t b,
                     std::size_t e) {
  lib::drlog.debug("{}\n", title);

  auto op = [](auto n) { return n * n; };
  int iota_base = 100;

  lib::distributed_vector<int> dvi1(n), dvr1(n);
  rng::iota(dvi1, iota_base);
  dvi1.fence();
  lib::transform(dvi1.begin() + b, dvi1.begin() + e, dvr1.begin(), op);
  dvr1.fence();
  lib::drlog.debug("dvr1 {}\n", dvr1);

  lib::distributed_vector<int> dvi2(n), dvr2(n);
  rng::iota(dvi2, iota_base);
  dvi2.fence();

  if (comm_rank == 0) {
    std::transform(dvi2.begin() + b, dvi2.begin() + e, dvr2.begin(), op);

    std::vector<int> v(n), vr(n);
    rng::iota(v, iota_base);
    std::transform(v.begin() + b, v.begin() + e, vr.begin(), op);
    expect_eq(vr, dvr1);
    expect_eq(vr, dvr2);
  }
}

TEST(CpuMpiTests, TransformDistributedVector) {
  std::size_t n = 10;

  check_transform("full vector", n, 0, n);
  check_transform("partial vector", n, n / 2 - 1, n / 2 + 1);
}

void check_transform2(std::string title, std::size_t n, std::size_t b,
                      std::size_t e) {
  lib::drlog.debug("{}\n", title);

  auto op = [](auto n, auto m) { return n * m; };
  int iota_base1 = 100;
  int iota_base2 = 1000;

  lib::distributed_vector<int> dvi1_1(n), dvi2_1(n), dvr_1(n);
  rng::iota(dvi1_1, iota_base1);
  rng::iota(dvi2_1, iota_base2);
  dvi1_1.fence();
  dvi2_1.fence();
  lib::transform(dvi1_1.begin() + b, dvi1_1.begin() + e, dvi2_1.begin() + b,
                 dvr_1.begin() + b, op);

  lib::distributed_vector<int> dvi1_2(n), dvi2_2(n), dvr_2(n);
  rng::iota(dvi1_2, iota_base1);
  rng::iota(dvi2_2, iota_base2);
  dvi1_2.fence();
  dvi2_2.fence();

  if (comm_rank == 0) {
    std::transform(dvi1_2.begin() + b, dvi1_2.begin() + e, dvi2_2.begin() + b,
                   dvr_2.begin() + b, op);

    std::vector<int> v1(n), v2(n), vr(n);
    rng::iota(v1, iota_base1);
    rng::iota(v2, iota_base2);
    std::transform(v1.begin() + b, v1.begin() + e, v2.begin() + b,
                   vr.begin() + b, op);
    expect_eq(vr, dvr_1);
    expect_eq(vr, dvr_2);
  }
}

TEST(CpuMpiTests, TransformDistributedVector2) {
  std::size_t n = 10;

  check_transform("full vector", n, 0, n);
  check_transform("partial vector", n, n / 2 - 1, n / 2 + 1);
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

  dv_in.halo().exchange_begin();
  dv_in.halo().exchange_finalize();
  lib::transform(dv_in.begin() + 1, dv_in.end() - 1, dv_out2.begin() + 1, op);
  if (comm_rank == 0) {
    expect_eq(dv_out2, v_out);
  }

  fmt::print("Initial:   {}\n", v_in);
  fmt::print("Reference: {}\n", v_out);
  fmt::print("Test:      {}\n", dv_out1);
  fmt::print("Test2:      {}\n", dv_out2);
}
