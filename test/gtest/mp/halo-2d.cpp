// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xp-tests.hpp"

TEST(Halo2D, exchange_2d_test) {
  dr::mp::distributed_mdarray<int, 2> dv({10, 10}, dr::mp::distribution().halo(1));

  DRLOG("exchange start");
  dv.halo().exchange();
  DRLOG("exchange end");
}
