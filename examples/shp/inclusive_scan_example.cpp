// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <dr/shp.hpp>
#include <sycl/sycl.hpp>

#include <iostream>

#include <fmt/core.h>
#include <fmt/ranges.h>

int main(int argc, char **argv) {
  printf("Creating NUMA devices...\n");
  auto devices = dr::shp::get_numa_devices(sycl::default_selector_v);
  dr::shp::init(devices);

  for (auto &device : devices) {
    std::cout << "  Device: " << device.get_info<sycl::info::device::name>()
              << "\n";
  }

  dr::shp::distributed_vector<int, dr::shp::device_allocator<int>> v(100);

  std::vector<int> lv(100);

  std::iota(lv.begin(), lv.end(), 0);
  dr::shp::copy(lv.begin(), lv.end(), v.begin());

  fmt::print(" v: {}\n", v);
  fmt::print("lv: {}\n", lv);

  std::inclusive_scan(lv.begin(), lv.end(), lv.begin());
  dr::shp::inclusive_scan(dr::shp::par_unseq, v, v);

  fmt::print(" (after)  v: {}\n", v);
  fmt::print(" (after) lv: {}\n", lv);

  for (std::size_t i = 0; i < lv.size(); i++) {
    int x = lv[i];
    int y = v[i];
    if (x != y) {
      printf("(%lu) %d != %d\n", i, x, y);
    }
  }

  std::iota(lv.begin(), lv.end(), 0);
  dr::shp::copy(lv.begin(), lv.end(), v.begin());

  dr::shp::distributed_vector<int, dr::shp::device_allocator<int>> o(v.size() +
                                                                     100);

  std::inclusive_scan(lv.begin(), lv.end(), lv.begin(), std::plus<>(), 12);
  dr::shp::inclusive_scan(dr::shp::par_unseq, v, o, std::plus<>(), 12);

  fmt::print(" (after)  v: {}\n",
             rng::subrange(o.begin(), o.begin() + v.size()));
  fmt::print(" (after) lv: {}\n", lv);

  for (std::size_t i = 0; i < lv.size(); i++) {
    int x = lv[i];
    int y = o[i];
    if (x != y) {
      printf("(%lu) %d != %d\n", i, x, y);
    }
  }

  return 0;
}
