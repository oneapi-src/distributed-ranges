#include "sycl-mpi-tests.hpp"

using T = int;
using Alloc = lib::sycl_shared_allocator<T>;
using DV = lib::distributed_vector<T, Alloc>;

void check_reduce(std::size_t n, std::size_t b, std::size_t e) {
  static_assert(lib::sycl_distributed_contiguous_iterator<DV::iterator>);
  Alloc alloc;

  auto op = std::plus<>();
  T init = 10000;
  T iota_base = 100;
  int root = 0;

  DV dv1(alloc, n);
  rng::iota(dv1, iota_base);
  dv1.fence();
  auto dv1_sum = lib::reduce(root, dv1.begin() + b, dv1.begin() + e, init, op);

  DV dv2(alloc, n);
  rng::iota(dv2, iota_base);
  dv2.fence();

  if (comm_rank == 0) {
    auto dv2_sum = std::reduce(dv2.begin() + b, dv2.begin() + e, init, op);

    std::vector<T> v(n);
    rng::iota(v, iota_base);
    auto v_sum = std::reduce(v.begin() + b, v.begin() + e, init, op);
    EXPECT_EQ(v_sum, dv1_sum);
    EXPECT_EQ(v_sum, dv2_sum);
  }
}

TEST(SyclMpiTests, ReduceDistributedVector) {
  std::size_t n = 10;

  check_reduce(n, 0, n);
  check_reduce(n, n / 2 - 1, n / 2 + 1);
}
