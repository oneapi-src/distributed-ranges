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

TEST(CpuMpiTests, Mdspan) {
  using T = double;

  std::vector<T> v(n);

  stdex::mdspan<T, dyn_2d, stdex::layout_right> m{v.data(), rows, cols};
  check_mdspan(v, m);

  using dvector = lib::distributed_vector<T>;
  dvector dv(n);
  dv.fence();

  stdex::mdspan<double, dyn_2d, stdex::layout_right,
                lib::distributed_accessor<dvector>>
      dm{dv.begin(), rows, cols};
  check_mdspan(dv, dm);
  dv.fence();
}

TEST(CpuMpiTests, Mdarray) {
  using T = double;

  stdex::mdarray<T, dyn_2d> m(rows, cols);
  stdex::mdarray<T, dyn_2d> mT(cols, rows);
  // lib::transpose(m, mT);
}
