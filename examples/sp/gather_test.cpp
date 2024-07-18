// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <dr/sp.hpp>
#include <fmt/core.h>

namespace sp = dr::sp;

int main(int argc, char **argv) {
  auto devices = dr::sp::get_numa_devices(sycl::default_selector_v);
  sp::init(devices);

  using T = int;

  std::size_t n_elements = 2 * 1000ull * 1000 * 1000ull;

  if (argc == 2) {
    n_elements = std::atoll(argv[1]);
  }

  fmt::print("Transfer size {} GB\n", n_elements * 1e-9);

  sp::distributed_vector<T> d_vec(n_elements);

  sp::fill(d_vec.begin(), d_vec.end(), 12);

  using vector_type = sp::vector<T, sp::device_allocator<T>>;
  std::vector<vector_type> local_data;

  fmt::print("Allocating...\n");
  for (std::size_t i = 0; i < sp::nprocs(); i++) {
    sp::device_allocator<T> allocator(sp::context(), sp::devices()[i]);
    local_data.emplace_back(n_elements, allocator);
  }

  fmt::print("Gather BW tests...\n");
  for (std::size_t i = 0; i < sp::nprocs(); i++) {
    auto &&local = local_data[i];

    auto begin = std::chrono::high_resolution_clock::now();
    sp::copy(d_vec.begin(), d_vec.end(), local.begin());
    auto end = std::chrono::high_resolution_clock::now();

    double duration = std::chrono::duration<double>(end - begin).count();

    std::size_t n_bytes = n_elements * sizeof(T);
    double n_gbytes = double(n_bytes) * 1e-9;

    double bw = n_gbytes / duration;

    fmt::print("Gather to -> {}. {} GB/s\n", i, bw);
  }

  fmt::print("Scatter BW tests...\n");
  for (std::size_t i = 0; i < sp::nprocs(); i++) {
    auto &&local = local_data[i];

    auto begin = std::chrono::high_resolution_clock::now();
    sp::copy(local.begin(), local.end(), d_vec.begin());
    auto end = std::chrono::high_resolution_clock::now();

    double duration = std::chrono::duration<double>(end - begin).count();

    std::size_t n_bytes = n_elements * sizeof(T);
    double n_gbytes = double(n_bytes) * 1e-9;

    double bw = n_gbytes / duration;

    fmt::print("Gather to -> {}. {} GB/s\n", i, bw);
  }

  fmt::print("Allgather BW tests...\n");
  auto begin = std::chrono::high_resolution_clock::now();
  std::vector<sycl::event> events;

  for (std::size_t i = 0; i < sp::nprocs(); i++) {
    auto &&local = local_data[i];
    auto e = sp::copy_async(d_vec.begin(), d_vec.end(), local.begin());
    events.push_back(e);
  }

  sycl::event::wait(events);
  auto end = std::chrono::high_resolution_clock::now();
  events.clear();

  double duration = std::chrono::duration<double>(end - begin).count();

  std::size_t n_bytes = n_elements * sp::nprocs() * sizeof(T);
  double n_gbytes = double(n_bytes) * 1e-9;

  double bw = n_gbytes / duration;

  fmt::print("Allgather. {} GB/s\n", bw);

  fmt::print("Allgather balanced BW tests...\n");
  begin = std::chrono::high_resolution_clock::now();

  for (std::size_t i = 0; i < sp::nprocs(); i++) {
    auto &&segments = d_vec.segments();
    auto &&local = local_data[i];
    for (std::size_t j_ = 0; j_ != segments.size(); j_++) {
      std::size_t j = (j_ + i) % std::size_t(segments.size());

      auto &&segment = segments[j];

      std::size_t offset = j * segments[0].size();

      auto e = sp::copy_async(segment.begin(), segment.end(),
                              local.begin() + offset);
      events.push_back(e);
    }
  }

  sycl::event::wait(events);
  end = std::chrono::high_resolution_clock::now();

  duration = std::chrono::duration<double>(end - begin).count();

  n_bytes = n_elements * sp::nprocs() * sizeof(T);
  n_gbytes = double(n_bytes) * 1e-9;

  bw = n_gbytes / duration;

  fmt::print("Allgather. {} GB/s\n", bw);

  return 0;
}
