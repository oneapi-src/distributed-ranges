// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <CL/sycl.hpp>
#include <dr/shp/shp.hpp>

#include <iostream>

#include <dr/shp/algorithms/inclusive_scan.hpp>
#include <fmt/core.h>
#include <fmt/ranges.h>

int main(int argc, char **argv) {
  namespace sycl = cl::sycl;

  printf("Creating NUMA devices...\n");
  auto devices = shp::get_duplicated_devices(sycl::default_selector_v, 8);
  shp::init(devices);

  for (auto &device : devices) {
    std::cout << "  Device: " << device.get_info<sycl::info::device::name>()
              << "\n";
  }

  shp::distributed_vector<int, shp::device_allocator<int>> v(100);

  shp::for_each(shp::par_unseq, shp::enumerate(v), [](auto &&tuple) {
    auto &&[idx, value] = tuple;
    value = idx;
  });

  std::vector<int> lv(100);

  shp::copy(v.begin(), v.end(), lv.begin());

  fmt::print("local v: {}\n", lv);

  std::iota(v.begin(), v.end(), 0);
  fmt::print("(before) v: {}\n", v);

  shp::inclusive_scan(shp::par_unseq, v);

  fmt::print(" (after) v: {}\n", v);

  for (auto &&seg : v.segments()) {
    fmt::print("Rank {}: {}\n", seg.rank(), seg);
  }

  std::inclusive_scan(lv.begin(), lv.end(), lv.begin());

  for (size_t i = 0; i < v.size(); i++) {
    int x = v[i];
    int y = lv[i];
    if (x != y) {
      printf("%d != %d\n", x, y);
    }
    assert(x == y);
  }

  shp::distributed_vector<int> o(v.size() + 100);

  std::iota(v.begin(), v.end(), 0);

  shp::inclusive_scan(shp::par_unseq, v, o);

  fmt::print("o: {}\n", rng::subrange(o.begin(), o.begin() + v.size()));

  for (size_t i = 0; i < lv.size(); i++) {
    int x = lv[i];
    int y = o[i];
    if (x != y) {
      printf("(%lu) %d != %d\n", i, x, y);
    }
  }

  return 0;
}
