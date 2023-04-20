// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <dr/shp.hpp>

template <dr::distributed_iterator Iter> void iter(Iter) {}

int main(int argc, char **argv) {
  printf("Creating NUMA devices...\n");
  auto devices = dr::shp::get_numa_devices(sycl::default_selector_v);
  dr::shp::init(devices);

  for (auto &device : devices) {
    std::cout << "  Device: " << device.get_info<sycl::info::device::name>()
              << "\n";
  }

  dr::shp::distributed_vector<int, dr::shp::device_allocator<int>> v(100);

  dr::shp::for_each(dr::shp::par_unseq, dr::shp::enumerate(v),
                    [](auto &&tuple) {
                      auto &&[idx, value] = tuple;
                      value = idx;
                    });

  dr::shp::for_each(dr::shp::par_unseq, v,
                    [](auto &&value) { value = value + 2; });

  std::size_t sum = dr::shp::reduce(dr::shp::par_unseq, v, int(0), std::plus{});

  dr::shp::print_range(v);

  std::cout << "Sum: " << sum << std::endl;

  std::vector<int> local_vec(v.size());
  std::iota(local_vec.begin(), local_vec.end(), 0);

  dr::shp::print_range(local_vec, "local vec");

  dr::shp::copy(local_vec.begin(), local_vec.end(), v.begin());

  dr::shp::print_range(v, "vec after copy");

  dr::shp::for_each(dr::shp::par_unseq, v,
                    [](auto &&value) { value = value + 2; });

  dr::shp::print_range(v, "vec after update");

  dr::shp::copy(v.begin(), v.end(), local_vec.begin());

  dr::shp::print_range(local_vec, "local vec after copy");

  return 0;
}
