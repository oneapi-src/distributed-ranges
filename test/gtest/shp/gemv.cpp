// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "shp-tests.hpp"
#include <dr/shp/shp.hpp>
#include <fmt/core.h>
#include <fmt/ranges.h>

// hard to reproduce fails
TEST(SparseMatrix, DISABLED_Gemv) {
  size_t m = 100;
  size_t k = 100;

  shp::sparse_matrix<float> a(
      {m, k}, 0.1f,
      shp::block_cyclic({shp::tile::div, shp::tile::div}, {shp::nprocs(), 1}));

  shp::distributed_vector<float> b(k, 1.f);
  shp::distributed_vector<float> c(m, 0.f);

  shp::gemv(c, a, b);

  std::vector<float> c_local(m);

  shp::copy(c.begin(), c.end(), c_local.begin());

  std::vector<float> c_ref(m, 0.f);

  for (auto &&[index, v] : a) {
    auto &&[i, k] = index;

    c_ref[i] += v;
  }

  EXPECT_EQ(c_ref, c_local);
}
