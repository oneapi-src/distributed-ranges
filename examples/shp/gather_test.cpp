// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <dr/shp.hpp>
#include <fmt/core.h>

namespace shp = dr::shp;

int main(int argc, char **argv) {
  auto devices = dr::shp::get_numa_devices(sycl::default_selector_v);
  shp::init(devices);

  using T = int;

  std::size_t n_elements = 2 * 1000ull * 1000 * 1000ull;

  shp::distributed_vector<T> d_vec(n_elements);

  shp::fill(d_vec.begin(), d_vec.end(), 12);

  using vector_type = shp::vector<T, shp::device_allocator<T>>;
  std::vector<vector_type> local_data;

  fmt::print("Allocating...\n");
  for (std::size_t i = 0; i < shp::nprocs(); i++) {
    shp::device_allocator<T> allocator(shp::context(), shp::devices()[i]);
    local_data.emplace_back(n_elements, allocator);
  }

  fmt::print("BW tests...\n");
  for (std::size_t i = 0; i < shp::nprocs(); i++) {
    auto &&local = local_data[i];

    auto begin = std::chrono::high_resolution_clock::now();
    shp::copy(d_vec.begin(), d_vec.end(), local.begin());
    auto end = std::chrono::high_resolution_clock::now();

    double duration = std::chrono::duration<double>(end - begin).count();

    std::size_t n_bytes = n_elements * sizeof(T);
    double n_gbytes = double(n_bytes) * 1e-9;

    double bw = n_gbytes / duration;

    fmt::print("Gather to -> {}. {} GB/s\n", i, bw);
  }

  return 0;
}
