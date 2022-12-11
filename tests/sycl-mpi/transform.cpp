#include "sycl-mpi-tests.hpp"

using T = int;
using Alloc = lib::sycl_shared_allocator<T>;
using DV = lib::distributed_vector<T, Alloc>;

void check_transform(std::string title, std::size_t n, std::size_t b,
                     std::size_t e) {
  lib::drlog.debug("{}\n", title);

  Alloc alloc;
  auto op = [](auto n) { return n * n; };
  int iota_base = 100;

  DV dvi1(alloc, n), dvr1(alloc, n);
  rng::iota(dvi1, iota_base);
  dvi1.fence();
  lib::transform(dvi1.begin() + b, dvi1.begin() + e, dvr1.begin(), op);
  dvr1.fence();
  lib::drlog.debug("dvr1 {}\n", dvr1);

  DV dvi2(alloc, n), dvr2(alloc, n);
  rng::iota(dvi2, iota_base);
  dvi2.fence();

  if (comm_rank == 0) {
    std::transform(dvi2.begin() + b, dvi2.begin() + e, dvr2.begin(), op);

    std::vector<int> v(n), vr(n);
    rng::iota(v, iota_base);
    std::transform(v.begin() + b, v.begin() + e, vr.begin(), op);
    EXPECT_TRUE(equal(vr, dvr1));
    EXPECT_TRUE(equal(vr, dvr2));
  }
}

TEST(SyclMpiTests, TransformDistributedVector) {
  std::size_t n = 10;

  check_transform("full vector", n, 0, n);
  check_transform("partial vector", n, n / 2 - 1, n / 2 + 1);
}

TEST(SyclMpiTests, Stencil) {
  std::size_t n = 10;
  auto op = [](auto &&v) {
    auto p = &v;
    return p[-1] + p[0] + p[+1];
  };

  std::vector<int> v_in(n), v_out(n);
  Alloc alloc;
  lib::stencil<1> s(1);
  DV dv_in(s, alloc, n), dv_out1(s, alloc, n), dv_out2(s, alloc, n);

  if (comm_rank == 0) {
    rng::iota(v_in, 100);
    rng::copy(v_in, dv_in.begin());
  }
  dv_in.fence();

  if (comm_rank == 0) {
    rng::transform(v_in.begin() + 1, v_in.end() - 1, v_out.begin() + 1, op);
    rng::transform(dv_in.begin() + 1, dv_in.end() - 1, dv_out1.begin() + 1, op);
    EXPECT_TRUE(unary_check(v_in, v_out, dv_out1));
  }

  lib::transform(dv_in.begin() + 1, dv_in.end() - 1, dv_out2.begin() + 1, op);
  if (comm_rank == 0) {
    EXPECT_TRUE(unary_check(v_in, v_out, dv_out2));
  }
}
