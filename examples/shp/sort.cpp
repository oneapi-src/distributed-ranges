// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <dr/shp.hpp>

#include <oneapi/dpl/algorithm>

#include <fmt/core.h>
#include <fmt/ranges.h>

namespace shp = dr::shp;

int main(int argc, char **argv) {
  auto devices = shp::get_duplicated_devices(sycl::default_selector_v, 4);
  shp::init(devices);

  std::size_t n = 100;

  shp::distributed_vector<int> v(n);

  srand48(time(0));

  for (std::size_t i = 0; i < v.size(); i++) {
    v[i] = lrand48() % 1000;
  }

  sort(v);

  fmt::print("v: {}\n", v);

  return 0;
}
