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

  dr::shp::distributed_vector<int, dr::shp::device_allocator<int>> v(100);

  dr::shp::for_each(dr::shp::par_unseq, dr::shp::enumerate(v),
                    [](auto &&tuple) {
                      auto &&[idx, value] = tuple;
                      value = idx;
                    });

  dr::shp::for_each(dr::shp::par_unseq, v, [](auto &&value) { value += 2; });

  dr::shp::print_range(v, "Distributed vector");

  // Create trimmed view.
  // `trimmed_view` is a distributed range.
  auto trimmed_view = dr::shp::views::take(v, 53);
  dr::shp::print_range(trimmed_view, "Trimmed View");

  auto sum = dr::shp::reduce(dr::shp::par_unseq, v, 0, std::plus{});
  std::cout << "Total sum: " << sum << std::endl;

  auto tsum = dr::shp::reduce(dr::shp::par_unseq, trimmed_view, 0, std::plus{});
  std::cout << "Trimmed sum: " << tsum << std::endl;

  dr::shp::print_range(v | rng::views::drop(40) |
                       dr::shp::views::slice({5, 10}));

  return 0;
}
