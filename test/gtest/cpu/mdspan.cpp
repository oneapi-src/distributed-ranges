// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <optional>

#include "cpu-tests.hpp"

// using rows as divideable by 2,3,4... because tests below assume storage_size
// without redundancy
const std::size_t rows = 24, cols = 38, n = rows * cols;
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

  using dspan = lib::distributed_mdspan<T, dyn_2d>;
  assert_distributed_range<dspan>();
  dspan dm(dv, rows, cols);

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

  if (comm_rank == 0) {
    dm(4, 1) = 9;
  }
  dm.fence();
  EXPECT_EQ(dm(4, 1), 9);
}

TEST(CpuMpiTests, distributed_mdspan_local) {
  using T = double;

  std::vector<T> v(n);

  using dvector = lib::distributed_vector<T>;
  dvector dv(n);

  using dspan = lib::distributed_mdspan<T, dyn_2d>;
  dspan dm(dv, rows, cols);

  auto local = dm.local();
  local(0, 0) = 99;
  dm.fence();

  EXPECT_EQ(local(0, 0), 99);
  EXPECT_EQ(dm(0, 0), 99);
}

void check_mdarray(auto &m) {
  if (comm_rank != 0)
    return;
  std::fill(m.begin(), m.end(), 9);
  EXPECT_EQ(m(1, 2), 9);
  m(1, 2) = 3;
  EXPECT_EQ(m(1, 2), 3);
}

TEST(CpuMpiTests, distributed_mdarray_basic) {
  using T = double;

  using dmatrix = lib::distributed_mdarray<T, dyn_2d>;
  assert_distributed_range<dmatrix>();

  dmatrix dm(rows, cols);
  check_mdarray(dm);

  dmatrix dm2(rows, cols);
  dm2(1, 2) = 99;
  dm(2, 1) = dm2(1, 2);
  EXPECT_EQ(dm(2, 1), 99);
  EXPECT_EQ(dm2(1, 2), 99);
}

TEST(CpuMpiTests, distributed_mdarray_local) {
  using T = double;

  using dmatrix = lib::distributed_mdarray<T, dyn_2d>;
  dmatrix dm(rows, cols);

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
}

namespace {
template <typename T> void fill_array(T &arr, size_t rows, size_t cols) {
  for (std::size_t i = 0; i < rows; i++) {
    for (std::size_t j = 0; j < cols; j++) {
      arr(i, j) = i * 100 + j;
    }
  }
}
} // namespace

// Disabled because random fails in CI
TEST(CpuMpiTests, DISABLED_distributed_mdarray_transpose) {
  using T = double;

  auto test_distributed_mdarray_transpose = [&](std::size_t rows,
                                                std::size_t cols) {
    using dmatrix = lib::distributed_mdarray<T, dyn_2d>;
    dmatrix dsrc(rows, cols);
    dmatrix ddst(cols, rows);

    if (comm_rank == 0) {
      fill_array(dsrc, rows, cols);
    }
    dsrc.fence();

    lib::collective::transpose(dsrc, ddst);
    ddst.fence();

    if (comm_rank == 0) {
      using lmatrix = stdex::mdarray<T, dyn_2d>;
      lmatrix lsrc(rows, cols);
      fill_array(lsrc, rows, cols);

      lmatrix ldst(cols, rows);
      lib::collective::transpose(lsrc, ldst);

      EXPECT_EQ(lsrc(1, 2), ldst(2, 1));
      EXPECT_EQ(lsrc(1, 2), dsrc(1, 2));
      EXPECT_EQ(ldst(1, 2), ddst(1, 2));
      expect_eq(lsrc, dsrc);
      expect_eq(ldst, ddst);
    }
  };

  for (size_t r = 3; r <= rows; r++)
    for (size_t c = 3; c <= cols; c++)
      test_distributed_mdarray_transpose(r, c);
}

TEST(CpuMpiTests, distribute_to_local_transpose) {
  using T = double;
  std::vector<T> v;
  if (comm_rank == 0) {
    v.resize(n);
  }

  lib::distributed_mdarray<T, dyn_2d> dsrc(rows, cols);
  std::experimental::mdspan<T, dyn_2d> ldst(v.data(), cols, rows);

  if (comm_rank == 0)
    fill_array(dsrc, rows, cols);
  dsrc.fence();

  if (comm_rank == 0) {
    lib::collective::transpose(0, dsrc, std::make_optional(ldst));
  } else {
    lib::collective::transpose(
        0, dsrc, std::optional<std::experimental::mdspan<T, dyn_2d>>());
  }

  if (comm_rank == 0) {
    EXPECT_EQ(dsrc(0, 0), ldst(0, 0));
    EXPECT_EQ(dsrc(1, 0), ldst(0, 1));
    EXPECT_EQ(dsrc(rows - 1, cols - 2), ldst(cols - 2, rows - 1));
  }
}

TEST(CpuMpiTests, local_to_distribute_transpose) {
  using T = double;
  std::vector<T> v;
  if (comm_rank == 0) {
    v.resize(n);
  }

  std::experimental::mdspan<T, dyn_2d> lsrc(v.data(), rows, cols);
  lib::distributed_mdarray<T, dyn_2d> ddst(cols, rows);

  if (comm_rank == 0)
    fill_array(lsrc, rows, cols);

  if (comm_rank == 0) {
    lib::collective::transpose(0, std::make_optional(lsrc), ddst);
  } else {
    lib::collective::transpose(
        0, std::optional<std::experimental::mdspan<T, dyn_2d>>(), ddst);
  }

  if (comm_rank == 0) {
    EXPECT_EQ(lsrc(0, 0), ddst(0, 0));
    EXPECT_EQ(lsrc(1, 0), ddst(0, 1));
    EXPECT_EQ(lsrc(rows - 1, cols - 2), ddst(cols - 2, rows - 1));
  }
}
