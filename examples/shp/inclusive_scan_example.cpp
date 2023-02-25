// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <CL/sycl.hpp>
#include <dr/shp/shp.hpp>

#include <iostream>

#include <fmt/core.h>
#include <fmt/ranges.h>

int main(int argc, char **argv) {
  namespace sycl = cl::sycl;

  printf("Creating NUMA devices...\n");
  auto devices = shp::get_devices(sycl::gpu_selector_v);
  shp::init(devices);

  for (auto &device : devices) {
    std::cout << "  Device: " << device.get_info<sycl::info::device::name>()
              << "\n";
  }

  shp::distributed_vector<int, shp::device_allocator<int>> v(100);

  std::vector<int> lv(100);

  std::iota(lv.begin(), lv.end(), 0);
  shp::copy(lv.begin(), lv.end(), v.begin());


  fmt::print(" v: {}\n", v);
  fmt::print("lv: {}\n", lv);

  std::inclusive_scan(lv.begin(), lv.end(), lv.begin());
  shp::inclusive_scan(shp::par_unseq, v, v);

  fmt::print(" (after)  v: {}\n", v);
  fmt::print(" (after) lv: {}\n", lv);

  for (size_t i = 0; i < lv.size(); i++) {
    int x = lv[i];
    int y = v[i];
    if (x != y) {
      printf("(%lu) %d != %d\n", i, x, y);
    }
  }

  std::iota(lv.begin(), lv.end(), 0);
  shp::copy(lv.begin(), lv.end(), v.begin());

  shp::distributed_vector<int, shp::device_allocator<int>> o(v.size() + 100);

  std::inclusive_scan(lv.begin(), lv.end(), lv.begin());
  shp::inclusive_scan(shp::par_unseq, v, o);

  fmt::print(" (after)  v: {}\n", rng::subrange(o.begin(), o.begin() + v.size()));
  fmt::print(" (after) lv: {}\n", lv);

  for (size_t i = 0; i < lv.size(); i++) {
    int x = lv[i];
    int y = o[i];
    if (x != y) {
      printf("(%lu) %d != %d\n", i, x, y);
    }
  }

  return 0;
}
