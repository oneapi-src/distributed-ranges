#include "cpu-tests.hpp"

const std::size_t rows = 20, cols = 10, n = rows * cols;
using dyn_2d = stdex::dextents<std::size_t, 2>;

void check_mdspan(auto &v, auto &m) {
  if (comm_rank != 0)
    return;

  rng::fill(v, 9);
  EXPECT_EQ(v[3 * cols + 2], 9);
  EXPECT_EQ(m(3, 2), 9);

  m(3, 2) = 1;
  EXPECT_EQ(m(3, 2), 1);
  EXPECT_EQ(v[3 * cols + 2], 1);
}

TEST(CpuMpiTests, distributed_mdspan) {
  using T = double;

  std::vector<T> v(n);

  stdex::mdspan<T, dyn_2d> m{v.data(), rows, cols};
  check_mdspan(v, m);

  using dvector = lib::distributed_vector<T>;
  dvector dv(n);
  dv.fence();

  using dspan = lib::distributed_mdspan<T, dyn_2d>;
  assert_distributed_range<dspan>();
  dspan dm(dv, rows, cols);
  dm.fence();
  check_mdspan(dv, dm);
  dm.fence();
  dv.fence();
  if (comm_rank == 0) {
    std::fill(dm.begin(), dm.end(), 8);
  }
  dm.fence();
  if (comm_rank == 0) {
    EXPECT_EQ(dv[3 * cols + 2], 8);
  }
  EXPECT_EQ(dm(3, 2), 8);

  dv.fence();
}

void check_mdarray(auto &m) {
  if (comm_rank != 0)
    return;
  std::fill(m.begin(), m.end(), 9);
  EXPECT_EQ(m(1, 2), 9);
  m(1, 2) = 3;
  EXPECT_EQ(m(1, 2), 3);
}

TEST(CpuMpiTests, distributed_mdarray) {
  using T = double;

  using dmatrix = lib::distributed_mdarray<T, dyn_2d>;
  assert_distributed_range<dmatrix>();

  dmatrix dm(rows, cols);
  dm.fence();
  check_mdarray(dm);

  dm.fence();
}
