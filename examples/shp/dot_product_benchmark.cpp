// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <dr/shp/shp.hpp>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/numeric>

template <lib::distributed_range X, lib::distributed_range Y>
auto dot_product_distributed(X &&x, Y &&y) {
  auto z = shp::views::zip(x, y) | lib::views::transform([](auto &&elem) {
             auto &&[a, b] = elem;
             return a * b;
           });

  return shp::reduce(shp::par_unseq, z, 0, std::plus());
}

template <rng::forward_range X, rng::forward_range Y>
auto dot_product_onedpl(sycl::queue q, X &&x, Y &&y) {
  auto z = rng::views::zip(x, y) | lib::views::transform([](auto &&elem) {
             auto &&[a, b] = elem;
             return a * b;
           });
  oneapi::dpl::execution::device_policy policy(q);
  shp::__detail::direct_iterator d_first(z.begin());
  shp::__detail::direct_iterator d_last(z.end());
  return oneapi::dpl::experimental::reduce_async(
             policy, d_first, d_last, rng::range_value_t<X>(0), std::plus())
      .get();
}

template <rng::forward_range X, rng::forward_range Y>
auto dot_product_sequential(X &&x, Y &&y) {
  auto z = shp::views::zip(x, y) | rng::views::transform([](auto &&elem) {
             auto &&[a, b] = elem;
             return a * b;
           });

  return std::reduce(z.begin(), z.end(), rng::range_value_t<X>(0), std::plus());
}

template <lib::distributed_iterator Iter, std::integral T>
void iota(Iter begin, Iter end, T value) {
  auto r = rng::subrange(begin, end);

  auto iota_view = rng::views::iota(T(value), T(value + rng::distance(r)));

  shp::for_each(shp::par_unseq, shp::views::zip(iota_view, r),
                [=](auto &&elem) {
                  auto &&[idx, v] = elem;
                  v = idx;
                });
}

int main(int argc, char **argv) {
  auto devices = shp::get_numa_devices(sycl::default_selector_v);
  shp::init(devices);

  std::size_t n = 1ull * 1024 * 1024 * 1024ull;

  using T = int;

  std::vector<T> x_local(n);
  std::vector<T> y_local(n);
  std::iota(x_local.begin(), x_local.end(), 0);
  std::iota(y_local.begin(), y_local.end(), 0);

  auto v_serial = dot_product_sequential(x_local, y_local);

  shp::distributed_vector<T> x(n);
  shp::distributed_vector<T> y(n);

  iota(x.begin(), x.end(), 0);
  iota(y.begin(), y.end(), 0);

  std::size_t n_iterations = 10;

  std::vector<double> durations;
  durations.reserve(n_iterations);

  // Execute on all devices with SHP:
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

  fmt::print("SHP executing on {} devices:\n", devices.size());
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

  sycl::queue q(shp::context(), shp::devices()[0]);

  T *x_d = sycl::malloc_device<T>(n, shp::devices()[0], shp::context());
  T *y_d = sycl::malloc_device<T>(n, shp::devices()[0], shp::context());
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

  return 0;
}
