// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <dr/shp.hpp>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/async>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/numeric>

#ifdef USE_MKL
#include <oneapi/mkl.hpp>
#endif

template <dr::distributed_range X, dr::distributed_range Y>
auto dot_product_distributed(X &&x, Y &&y) {
  auto z = dr::shp::views::zip(x, y) | dr::views::transform([](auto &&elem) {
             auto &&[a, b] = elem;
             return a * b;
           });

  return dr::shp::reduce(dr::shp::par_unseq, z, rng::range_value_t<X>(0),
                         std::plus());
}

template <rng::forward_range X, rng::forward_range Y>
auto dot_product_onedpl(sycl::queue q, X &&x, Y &&y) {
  auto z = rng::views::zip(x, y) | dr::views::transform([](auto &&elem) {
             auto &&[a, b] = elem;
             return a * b;
           });
  oneapi::dpl::execution::device_policy policy(q);
  dr::__detail::direct_iterator d_first(z.begin());
  dr::__detail::direct_iterator d_last(z.end());
  return oneapi::dpl::experimental::reduce_async(
             policy, d_first, d_last, rng::range_value_t<X>(0), std::plus())
      .get();
}

template <rng::forward_range X, rng::forward_range Y>
auto dot_product_sequential(X &&x, Y &&y) {
  auto z = dr::shp::views::zip(x, y) | rng::views::transform([](auto &&elem) {
             auto &&[a, b] = elem;
             return a * b;
           });

  return std::reduce(z.begin(), z.end(), rng::range_value_t<X>(0), std::plus());
}

int main(int argc, char **argv) {

  if (argc != 3) {
    fmt::print("usage: ./dot_product_benchmark [n_devices] [n_elements]\n");
    return 1;
  }

  std::size_t n_devices = std::atoll(argv[1]);

  std::size_t n = std::atoll(argv[2]);

  // auto devices_ = dr::shp::get_numa_devices(sycl::default_selector_v);
  auto devices_ = dr::shp::get_devices(sycl::default_selector_v);

  // std::size_t n_devices = devices_.size();

  auto devices =
      dr::shp::trim_devices(devices_, std::min(n_devices, devices_.size()));

  dr::shp::init(devices);

  fmt::print("Running with {} devices, {} elements.\n", devices.size(), n);
  dr::shp::print_device_details(devices);

  // Note that parallel results will not match sequential
  // results for large sizes due to floating point addition
  // non-determinism.
  // This does not indicate the benchmark is failing.
  // std::size_t n = 4ull * 1000 * 1000 * 1000ull;

  using T = float;

  std::vector<T> x_local(n, 1);
  std::vector<T> y_local(n, 1);

  auto v_serial = dot_product_sequential(x_local, y_local);

  std::size_t n_iterations = 10;
  T sum = 0;

  std::vector<double> durations;
  durations.reserve(n_iterations);

  {
    fmt::print("Initializing distributed vector...\n");
    dr::shp::distributed_vector<T> x(n, 1);
    dr::shp::distributed_vector<T> y(n, 1);

    // Execute on all devices with SHP:
    fmt::print("Execute with SHP...\n");
    for (std::size_t i = 0; i < n_iterations; i++) {
      auto begin = std::chrono::high_resolution_clock::now();
      auto v = dot_product_distributed(x, y);
      auto end = std::chrono::high_resolution_clock::now();
      double duration = std::chrono::duration<double>(end - begin).count();
      durations.push_back(duration);
      if (v != v_serial) {
        fmt::print("{} != {}\n", v, v_serial);
      }
      // assert(v == v_serial);
      sum += v;
    }

    fmt::print("SHP executing on {} devices:\n", devices.size());
    fmt::print("Durations: {}\n",
               durations |
                   rng::views::transform([](auto &&x) { return x * 1000; }));

    std::sort(durations.begin(), durations.end());

    double median_duration = durations[durations.size() / 2];

    fmt::print("Median duration: {} ms\n", median_duration * 1000);

    fmt::print("Result: {}\n", sum);
  }

  // Execute on one device:
  durations.clear();

  sycl::queue q(dr::shp::context(), dr::shp::devices()[0]);
  sycl::queue q_i(dr::shp::context(), dr::shp::devices()[0],
                  {sycl::property::queue::in_order()});

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

    if (v != v_serial) {
      fmt::print("{} != {}\n", v, v_serial);
    }
    // See note above about large problem sizes.
    // assert(v == v_serial);
    sum += v;
  }

  fmt::print("oneDPL executing on one device:\n");
  fmt::print("Durations: {}\n", durations | rng::views::transform([](auto &&x) {
                                  return x * 1000;
                                }));

  std::sort(durations.begin(), durations.end());

  double median_duration = durations[durations.size() / 2];

  fmt::print("Median duration: {} ms\n", median_duration * 1000);

  fmt::print("Result: {}\n", sum);

  durations.clear();

#ifdef USE_MKL

  T *d_result = sycl::malloc_device<T>(1, q);
  sum = 0;

  for (std::size_t i = 0; i < n_iterations; i++) {
    auto begin = std::chrono::high_resolution_clock::now();
    oneapi::mkl::blas::row_major::dot(q, n, x_d, 1, y_d, 1, d_result).wait();
    T v;
    q.memcpy(&v, d_result, sizeof(T)).wait();
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end - begin).count();
    durations.push_back(duration);

    if (v != v_serial) {
      fmt::print("{} != {}\n", v, v_serial);
    }
    // assert(v == v_serial);
    sum += v;
  }

  fmt::print("oneMKL executing on one device:\n");
  fmt::print("Durations: {}\n", durations | rng::views::transform([](auto &&x) {
                                  return x * 1000;
                                }));

  std::sort(durations.begin(), durations.end());

  median_duration = durations[durations.size() / 2];

  fmt::print("Median duration: {} ms\n", median_duration * 1000);

  fmt::print("Result: {}\n", sum);
#endif

  return 0;
}
