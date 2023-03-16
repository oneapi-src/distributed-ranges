// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <dr/shp/shp.hpp>

int main(int argc, char **argv) {
  auto devices = shp::get_numa_devices(sycl::gpu_selector_v);
  shp::init(devices);

  for (auto &device : devices) {
    std::cout << "  Device: " << device.get_info<sycl::info::device::name>()
              << "\n";
  }

  shp::distributed_vector<int, shp::device_allocator<int>> b(100);

  shp::for_each(shp::par_unseq, shp::enumerate(b), [](auto &&tuple) {
    auto &&[idx, value] = tuple;
    value = 1;
  });

  shp::distributed_vector<int, shp::device_allocator<int>> c(100);

  shp::for_each(shp::par_unseq, c, [](auto &&v) { v = 0; });

  shp::sparse_matrix<int> a(
      {100, 100}, 0.01,
      shp::block_cyclic({shp::tile::div, shp::tile::div}, {shp::nprocs(), 1}));

  printf("a tiles: %lu x %lu\n", a.grid_shape()[0], a.grid_shape()[1]);

  shp::print_range(b, "b");

  shp::print_matrix(a, "a");

  shp::gemv(c, a, b);

  shp::print_range(c, "c");

  return 0;
}
