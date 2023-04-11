// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <sycl/sycl.hpp>

#include <dr/mhp.hpp>

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/async>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/numeric>

using T = int;

MPI_Comm comm;
std::size_t comm_rank;
std::size_t comm_size;

std::size_t n = 1ull * 1024 * 1024ull;
std::size_t n_iterations = 10;

auto dot_product_distributed(lib::distributed_range auto &&x,
                             lib::distributed_range auto &&y) {
  auto z = mhp::views::zip(x, y) | lib::views::transform([](auto &&elem) {
             auto &&[a, b] = elem;
             return a * b;
           });

  return mhp::reduce(mhp::device_policy(), z, 0, std::plus());
}

template <rng::forward_range X, rng::forward_range Y>
auto dot_product_onedpl(sycl::queue q, X &&x, Y &&y) {
  auto z = rng::views::zip(x, y) | lib::views::transform([](auto &&elem) {
             auto &&[a, b] = elem;
             return a * b;
           });
  oneapi::dpl::execution::device_policy policy(q);
  lib::__detail::direct_iterator d_first(z.begin());
  lib::__detail::direct_iterator d_last(z.end());
  return oneapi::dpl::experimental::reduce_async(
             policy, d_first, d_last, rng::range_value_t<X>(0), std::plus())
      .get();
}

template <rng::forward_range X, rng::forward_range Y>
auto dot_product_sequential(X &&x, Y &&y) {
  auto z = rng::views::zip(x, y) | rng::views::transform([](auto &&elem) {
             auto &&[a, b] = elem;
             return a * b;
           });

  return std::reduce(z.begin(), z.end(), rng::range_value_t<X>(0), std::plus());
}

void stats(auto &durations, auto &sum, auto v_serial, auto &x_local,
           auto &y_local) {
  fmt::print("MHP executing on {} ranks:\n", comm_size);
  fmt::print("Durations: {}\n", durations | rng::views::transform([](auto &&x) {
                                  return x * 1000;
                                }));

  std::sort(durations.begin(), durations.end());

  double median_duration = durations[durations.size() / 2];

  fmt::print("Median duration: {} ms\n", median_duration * 1000);
  fmt::print("Result: {}\n", sum);

  // Execute on one device:
  durations.clear();
  durations.reserve(n_iterations);

  sycl::queue q;

  T *x_d = sycl::malloc_device<T>(n, q);
  T *y_d = sycl::malloc_device<T>(n, q);
  q.memcpy(x_d, x_local.data(), n * sizeof(T)).wait();
  q.memcpy(y_d, y_local.data(), n * sizeof(T)).wait();

  sum = 0;
  for (std::size_t i = 0; i < n_iterations; i++) {
    std::span<T> x(x_d, n);
    std::span<T> y(y_d, n);
    auto begin = std::chrono::high_resolution_clock::now();
    auto v = dot_product_onedpl(q, x, y);
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end - begin).count();
    durations.push_back(duration);
    assert(v == v_serial);
    sum += v;
  }

  fmt::print("oneDPL executing on one device:\n");
  fmt::print("Durations: {}\n", durations | rng::views::transform([](auto &&x) {
                                  return x * 1000;
                                }));

  std::sort(durations.begin(), durations.end());

  median_duration = durations[durations.size() / 2];

  fmt::print("Median duration: {} ms\n", median_duration * 1000);

  fmt::print("Result: {}\n", sum);
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  comm = MPI_COMM_WORLD;
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  comm_rank = rank;
  comm_size = size;

  mhp::init();

  std::vector<T> x_local(n);
  std::vector<T> y_local(n);
  std::iota(x_local.begin(), x_local.end(), 0);
  std::iota(y_local.begin(), y_local.end(), 0);

  auto v_serial = dot_product_sequential(x_local, y_local);

  mhp::distributed_vector<T> x(n);
  mhp::distributed_vector<T> y(n);
  mhp::iota(x.begin(), x.end(), 0);
  mhp::iota(y.begin(), y.end(), 0);

  std::vector<double> durations;
  durations.reserve(n_iterations);

  // Execute on all devices with MHP:
  T sum = 0;
  for (std::size_t i = 0; i < n_iterations; i++) {
    auto begin = std::chrono::high_resolution_clock::now();
    auto v = dot_product_distributed(x, y);
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end - begin).count();
    durations.push_back(duration);
    assert(v == v_serial);
    sum += v;
  }

  if (comm_rank == 0) {
    stats(durations, sum, v_serial, x_local, y_local);
  }

  return 0;
}
