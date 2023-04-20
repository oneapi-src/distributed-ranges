// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <dr/shp.hpp>

int main(int argc, char **argv) {
  auto devices = dr::shp::get_numa_devices(sycl::gpu_selector_v);
  dr::shp::init(devices);

  for (auto &device : devices) {
    std::cout << "  Device: " << device.get_info<sycl::info::device::name>()
              << "\n";
  }

  using T = float;

  dr::shp::distributed_vector<T, dr::shp::device_allocator<T>> b(100);

  dr::shp::for_each(dr::shp::par_unseq, dr::shp::enumerate(b),
                    [](auto &&tuple) {
                      auto &&[idx, value] = tuple;
                      value = 1;
                    });

  dr::shp::distributed_vector<T, dr::shp::device_allocator<T>> c(100);

  dr::shp::for_each(dr::shp::par_unseq, c, [](auto &&v) { v = 0; });

  dr::shp::sparse_matrix<T> a(
      {100, 100}, 0.01,
      dr::shp::block_cyclic({dr::shp::tile::div, dr::shp::tile::div},
                            {dr::shp::nprocs(), 1}));

  printf("a tiles: %lu x %lu\n", a.grid_shape()[0], a.grid_shape()[1]);

  dr::shp::print_range(b, "b");

  dr::shp::print_matrix(a, "a");

  dr::shp::gemv(c, a, b);

  dr::shp::print_range(c, "c");

  return 0;
}
