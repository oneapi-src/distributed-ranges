// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <dr/sp.hpp>

namespace sp = dr::sp;

int main(int argc, char **argv) {
  auto devices = sp::get_numa_devices(sycl::gpu_selector_v);
  sp::init(devices);

  for (auto &device : devices) {
    std::cout << "  Device: " << device.get_info<sycl::info::device::name>()
              << "\n";
  }

  using T = float;

  sp::distributed_vector<T, sp::device_allocator<T>> b(100);

  sp::duplicated_vector<T> b_duplicated(100);

  sp::for_each(sp::par_unseq, sp::enumerate(b), [](auto &&tuple) {
    auto &&[idx, value] = tuple;
    value = 1;
  });

  sp::distributed_vector<T, sp::device_allocator<T>> c(100);

  sp::for_each(sp::par_unseq, c, [](auto &&v) { v = 0; });

  sp::sparse_matrix<T> a(
      {100, 100}, 0.01,
      sp::block_cyclic({sp::tile::div, sp::tile::div}, {sp::nprocs(), 1}));

  printf("a tiles: %lu x %lu\n", a.grid_shape()[0], a.grid_shape()[1]);

  sp::print_range(b, "b");

  sp::print_matrix(a, "a");

  sp::gemv(c, a, b, b_duplicated);

  sp::print_range(c, "c");

  return 0;
}
