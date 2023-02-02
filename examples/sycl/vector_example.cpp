// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <CL/sycl.hpp>
#include <dr/shp/shp.hpp>

#include <ranges>

#include <iostream>

template <lib::distributed_iterator Iter> void iter(Iter) {}

int main(int argc, char **argv) {
  namespace sycl = cl::sycl;

  printf("Creating NUMA devices...\n");
  auto devices = shp::get_numa_devices(sycl::default_selector_v);
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

  shp::for_each(shp::par_unseq, v, [](auto &&value) { value += 2; });

  size_t sum = shp::reduce(shp::par_unseq, v, int(0), std::plus{});

  shp::print_range(v);

  std::cout << "Sum: " << sum << std::endl;

  std::vector<int> local_vec(v.size());
  std::iota(local_vec.begin(), local_vec.end(), 0);

  shp::print_range(local_vec, "local vec");

  shp::copy(local_vec.begin(), local_vec.end(), v.begin());

  shp::print_range(v, "vec after copy");

  shp::for_each(shp::par_unseq, v, [](auto &&value) { value += 2; });

  shp::print_range(v, "vec after update");

  shp::copy(v.begin(), v.end(), local_vec.begin());

  shp::print_range(local_vec, "local vec after copy");

  return 0;
}
