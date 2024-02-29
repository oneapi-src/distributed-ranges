// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <dr/shp.hpp>
#include <fmt/ranges.h>

int main(int argc, char **argv) {
  printf("Creating NUMA devices...\n");
  dr::shp::init(sycl::default_selector_v);

  for (auto &device : dr::shp::devices()) {
    std::cout << "  Device: " << device.get_info<sycl::info::device::name>()
              << "\n";
  }

  fmt::print("First check...\n");
  dr::shp::check_queues();

  fmt::print("Initializing distributed vector...\n");
  dr::shp::distributed_vector<int, dr::shp::device_allocator<int>> v(100);

  fmt::print("Second check...\n");
  dr::shp::check_queues();

  fmt::print("For each...\n");
  dr::shp::for_each(dr::shp::par_unseq, dr::shp::enumerate(v),
                    [](auto &&tuple) {
                      auto &&[idx, value] = tuple;
                      value = idx;
                    });

  dr::shp::for_each(dr::shp::par_unseq, v,
                    [](auto &&value) { value = value + 2; });

  fmt::print("Third check...\n");
  dr::shp::check_queues();

  fmt::print("Reduce...\n");
  std::size_t sum = dr::shp::reduce(dr::shp::par_unseq, v, int(0), std::plus{});

  dr::shp::print_range(v);

  std::cout << "Sum: " << sum << std::endl;

  std::vector<int> local_vec(v.size());
  std::iota(local_vec.begin(), local_vec.end(), 0);

  fmt::print("Fourth check...\n");
  dr::shp::check_queues();

  dr::shp::print_range(local_vec, "local vec");

  fmt::print("Fourth Two check...\n");
  dr::shp::check_queues();

  dr::shp::copy(local_vec.begin(), local_vec.end(), v.begin());

  fmt::print("Fourth Three check...\n");
  dr::shp::check_queues();

  dr::shp::print_range(v, "vec after copy");

  dr::shp::for_each(dr::shp::par_unseq, v,
                    [](auto &&value) { value = value + 2; });

  dr::shp::print_range(v, "vec after update");

  fmt::print("Fourth One check...\n");
  dr::shp::check_queues();

  dr::shp::copy(v.begin(), v.end(), local_vec.begin());

  dr::shp::print_range(local_vec, "local vec after copy");

  v.resize(200);
  dr::shp::print_range(v, "resized to 200");

  v.resize(50);
  dr::shp::print_range(v, "resized to 50");

  fmt::print("Fifth check...\n");
  dr::shp::check_queues();

  fmt::print("Getting ready to finalize...\n");
  fflush(stdout);

  fmt::print("Check queues...\n");
  dr::shp::check_queues();

  fmt::print("Finalizing...\n");
  fflush(stdout);

  dr::shp::finalize();


  fmt::print("Exiting...\n");
  fflush(stdout);
  return 0;
}
