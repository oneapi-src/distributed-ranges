// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <dr/sp.hpp>

int main(int argc, char **argv) {
  auto devices = dr::sp::get_numa_devices(sycl::gpu_selector_v);
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

  dr::sp::for_each(dr::sp::par_unseq, v, [](auto &&value) { value += 2; });

  dr::sp::print_range(v, "Distributed vector");

  // Create trimmed view.
  // `trimmed_view` is a distributed range.
  auto trimmed_view = dr::sp::views::take(v, 53);
  dr::sp::print_range(trimmed_view, "Trimmed View");

  auto sum = dr::sp::reduce(dr::sp::par_unseq, v, 0, std::plus{});
  std::cout << "Total sum: " << sum << std::endl;

  auto tsum = dr::sp::reduce(dr::sp::par_unseq, trimmed_view, 0, std::plus{});
  std::cout << "Trimmed sum: " << tsum << std::endl;

  dr::sp::print_range(v | rng::views::drop(40) | dr::sp::views::slice({5, 10}));

  return 0;
}
