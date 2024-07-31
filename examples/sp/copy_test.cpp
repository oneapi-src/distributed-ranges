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

  using vector_type = sp::vector<T, sp::device_allocator<T>>;

  std::vector<vector_type> send_data;
  std::vector<vector_type> receive_data;

  fmt::print("Allocating...\n");
  for (std::size_t i = 0; i < sp::nprocs(); i++) {
    sp::device_allocator<T> allocator(sp::context(), sp::devices()[i]);
    send_data.emplace_back(n_elements, allocator);
    receive_data.emplace_back(n_elements, allocator);

    sp::fill(send_data.back().begin(), send_data.back().end(), i);
    sp::fill(receive_data.back().begin(), receive_data.back().end(), i);
  }

  fmt::print("BW tests...\n");
  for (std::size_t i = 0; i < sp::nprocs(); i++) {
    for (std::size_t j = 0; j < sp::nprocs(); j++) {

      auto &&send = send_data[i];
      auto &&receive = receive_data[j];

      auto begin = std::chrono::high_resolution_clock::now();
      sp::copy(send.begin(), send.end(), receive.begin());
      auto end = std::chrono::high_resolution_clock::now();

      double duration = std::chrono::duration<double>(end - begin).count();

      std::size_t n_bytes = n_elements * sizeof(T);
      double n_gbytes = double(n_bytes) * 1e-9;

      double bw = n_gbytes / duration;

      fmt::print("Copy from {} -> {}. {} GB/s\n", i, j, bw);
    }
  }

  fmt::print("BW tests with raw copy, sending queue...\n");
  for (std::size_t i = 0; i < sp::nprocs(); i++) {
    for (std::size_t j = 0; j < sp::nprocs(); j++) {

      auto &&send = send_data[i];
      auto &&receive = receive_data[j];

      auto begin = std::chrono::high_resolution_clock::now();
      auto &&q = sp::__detail::queue(i);
      q.memcpy(receive.begin().local(), send.begin().local(),
               sizeof(T) * send.size())
          .wait();
      auto end = std::chrono::high_resolution_clock::now();

      double duration = std::chrono::duration<double>(end - begin).count();

      std::size_t n_bytes = n_elements * sizeof(T);
      double n_gbytes = double(n_bytes) * 1e-9;

      double bw = n_gbytes / duration;

      fmt::print("Copy from {} -> {}. {} GB/s\n", i, j, bw);
    }
  }

  fmt::print("BW tests with raw copy, receiving queue...\n");
  for (std::size_t i = 0; i < sp::nprocs(); i++) {
    for (std::size_t j = 0; j < sp::nprocs(); j++) {

      auto &&send = send_data[i];
      auto &&receive = receive_data[j];

      auto begin = std::chrono::high_resolution_clock::now();
      auto &&q = sp::__detail::queue(j);
      q.memcpy(receive.begin().local(), send.begin().local(),
               sizeof(T) * send.size())
          .wait();
      auto end = std::chrono::high_resolution_clock::now();

      double duration = std::chrono::duration<double>(end - begin).count();

      std::size_t n_bytes = n_elements * sizeof(T);
      double n_gbytes = double(n_bytes) * 1e-9;

      double bw = n_gbytes / duration;

      fmt::print("Copy from {} -> {}. {} GB/s\n", i, j, bw);
    }
  }

  return 0;
}
