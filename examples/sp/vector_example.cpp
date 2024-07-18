// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <dr/sp.hpp>

int main(int argc, char **argv) {
  printf("Creating NUMA devices...\n");
  auto devices = dr::sp::get_numa_devices(sycl::default_selector_v);
  dr::sp::init(devices);

  for (auto &device : devices) {
    std::cout << "  Device: " << device.get_info<sycl::info::device::name>()
              << "\n";
  }

  dr::sp::distributed_vector<int, dr::sp::device_allocator<int>> v(100);

  dr::sp::for_each(dr::sp::par_unseq, dr::sp::enumerate(v), [](auto &&tuple) {
    auto &&[idx, value] = tuple;
    value = idx;
  });

  dr::sp::for_each(dr::sp::par_unseq, v,
                   [](auto &&value) { value = value + 2; });

  std::size_t sum = dr::sp::reduce(dr::sp::par_unseq, v, int(0), std::plus{});

  dr::sp::print_range(v);

  std::cout << "Sum: " << sum << std::endl;

  std::vector<int> local_vec(v.size());
  std::iota(local_vec.begin(), local_vec.end(), 0);

  dr::sp::print_range(local_vec, "local vec");

  dr::sp::copy(local_vec.begin(), local_vec.end(), v.begin());

  dr::sp::print_range(v, "vec after copy");

  dr::sp::for_each(dr::sp::par_unseq, v,
                   [](auto &&value) { value = value + 2; });

  dr::sp::print_range(v, "vec after update");

  dr::sp::copy(v.begin(), v.end(), local_vec.begin());

  dr::sp::print_range(local_vec, "local vec after copy");

  v.resize(200);
  dr::sp::print_range(v, "resized to 200");

  v.resize(50);
  dr::sp::print_range(v, "resized to 50");

  return 0;
}
