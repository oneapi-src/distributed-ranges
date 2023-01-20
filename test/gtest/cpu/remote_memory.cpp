// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu-tests.hpp"

TEST(CpuMpiTests, RemotePointerRequirements) {
  using RP = lib::remote_pointer<int>;

  static_assert(std::forward_iterator<RP>);
}
