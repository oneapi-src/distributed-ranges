// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <shp/shp.hpp>

template <lib::distributed_range R> void distributed(R &&) {}

template <lib::remote_range R> void remote(R &&) {}

int main(int argc, char **argv) {
  namespace sycl = cl::sycl;
  auto devices = shp::get_numa_devices(sycl::gpu_selector_v);
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

  shp::print_range(v, "Distributed vector");

  // Create trimmed view.
  // `trimmed_view` is a distributed range.
  auto trimmed_view = std::ranges::views::take(v, 53);
  shp::print_range(trimmed_view, "Trimmed View");

  auto sum = shp::reduce(shp::par_unseq, v, 0, std::plus{});
  std::cout << "Total sum: " << sum << std::endl;

  auto tsum = shp::reduce(shp::par_unseq, trimmed_view, 0, std::plus{});
  std::cout << "Trimmed sum: " << tsum << std::endl;

  return 0;
}
