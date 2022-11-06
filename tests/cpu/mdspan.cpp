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

TEST(CpuMpiTests, distributed_mdspan_local) {
  using T = double;

  std::vector<T> v(n);

  using dvector = lib::distributed_vector<T>;
  dvector dv(n);
  dv.fence();

  using dspan = lib::distributed_mdspan<T, dyn_2d>;
  dspan dm(dv, rows, cols);
  dm.fence();

  auto local = dm.local();
  local(0, 0) = 99;
  dm.fence();

  EXPECT_EQ(local(0, 0), 99);
  EXPECT_EQ(dm(0, 0), 99);

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

TEST(CpuMpiTests, distributed_mdarray_local) {
  using T = double;

  using dmatrix = lib::distributed_mdarray<T, dyn_2d>;
  dmatrix dm(rows, cols);
  dm.fence();

  auto local = dm.local();
  local(0, 0) = 100 + comm_rank;
  EXPECT_EQ(local(0, 0), 100 + comm_rank);
  dm.fence();
  EXPECT_EQ(dm(0, 0), 100 + 0);
  dm.fence();

  *local.data_handle() = 200 + comm_rank;
  EXPECT_EQ(local(0, 0), 200 + comm_rank);
  dm.fence();
  EXPECT_EQ(dm(0, 0), 200 + 0);

  dm.fence();
}

TEST(CpuMpiTests, transpose) {
  using T = double;

  using dmatrix = lib::distributed_mdarray<T, dyn_2d>;
  dmatrix dsrc(rows, cols);
  dmatrix ddst(cols, rows);
  dsrc.fence();
  ddst.fence();

  if (comm_rank == 0) {
    std::iota(dsrc.begin(), dsrc.end(), 10);
  }
  dsrc.fence();

  lib::collective::transpose(dsrc, ddst);
  ddst.fence();

  if (comm_rank == 0) {
    using lmatrix = stdex::mdarray<T, dyn_2d>;
    lmatrix lsrc(rows, cols);
    lmatrix ldst(cols, rows);
    lib::collective::transpose(lsrc, ldst);

    for (std::size_t i = 0; i < ldst.extents().extent(0); i++) {
      for (std::size_t j = 0; j < ldst.extents().extent(1); j++) {
        EXPECT_EQ(ldst(i, j), ddst(i, j));
      }
    }
    // expect_eq(lsrc, dsrc);
    // expect_eq(ldst, ddst);
  }
  dsrc.fence();
  ddst.fence();
}
