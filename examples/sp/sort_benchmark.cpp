// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <dr/sp.hpp>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/async>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/numeric>

template <dr::distributed_range X, typename Compare = std::less<>>
auto sort_distributed(X &&x, Compare comp = Compare()) {
  dr::sp::sort(x.begin(), x.end(), comp);
}

template <rng::random_access_range X, typename Compare = std::less<>>
auto sort_onedpl(sycl::queue q, X &&x, Compare comp = Compare()) {
  oneapi::dpl::execution::device_policy policy(q);
  dr::__detail::direct_iterator d_first(x.begin());
  dr::__detail::direct_iterator d_last(x.end());
  oneapi::dpl::experimental::sort_async(policy, d_first, d_last, comp).wait();
}

template <rng::forward_range X, typename Compare = std::less<>>
auto sort_sequential(X &&x, Compare comp = Compare()) {
  std::sort(x.begin(), x.end(), comp);
}

template <rng::forward_range X> void fill_random(X &&x) {
  for (auto &&value : x) {
    value = drand48() * 100;
  }
}

int main(int argc, char **argv) {
  auto devices_ = dr::sp::get_numa_devices(sycl::default_selector_v);
  // auto devices = dr::sp::trim_devices(devices_, 8);
  auto devices = devices_;
  dr::sp::init(devices);

  // Note that parallel results will not match sequential
  // results for large sizes due to floating point addition
  // non-determinism.
  // This does not indicate the benchmark is failing.
  std::size_t n = 32ull * 1000 * 1000ull;

  using T = float;

  std::vector<T> x_local(n, 1);
  fill_random(x_local);

  std::vector<T> v_serial = x_local;

  sort_sequential(v_serial);

  dr::sp::distributed_vector<T> x(n);

  std::size_t n_iterations = 10;

  std::vector<double> durations;
  durations.reserve(n_iterations);

  // Execute on all devices with SP:
  for (std::size_t i = 0; i < n_iterations; i++) {
    dr::sp::distributed_vector<T> v = x;

    auto begin = std::chrono::high_resolution_clock::now();
    sort_distributed(v);
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end - begin).count();
    durations.push_back(duration);

    std::vector<T> v_local(n);
    dr::sp::copy(v.begin(), v.end(), v_local.begin());

    for (std::size_t j = 0; j < x_local.size(); j++) {
      assert(v_local[j] != v_serial[j]);
    }
  }

  fmt::print("SP executing on {} devices:\n", devices.size());
  fmt::print("Durations: {}\n", durations | rng::views::transform([](auto &&x) {
                                  return x * 1000;
                                }));

  std::sort(durations.begin(), durations.end());

  double median_duration = durations[durations.size() / 2];

  fmt::print("Median duration: {} ms\n", median_duration * 1000);

  // Execute on one device:
  durations.clear();
  durations.reserve(n_iterations);

  sycl::queue q(dr::sp::context(), dr::sp::devices()[0]);

  T *x_d = sycl::malloc_device<T>(n, q);
  q.memcpy(x_d, x_local.data(), n * sizeof(T)).wait();

  T *v_d = sycl::malloc_device<T>(n, q);

  for (std::size_t i = 0; i < n_iterations; i++) {
    q.memcpy(v_d, x_d, n * sizeof(T)).wait();
    auto begin = std::chrono::high_resolution_clock::now();
    sort_onedpl(q, std::span<T>(v_d, n));
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end - begin).count();
    durations.push_back(duration);

    std::vector<T> v(n);
    q.memcpy(v.data(), v_d, sizeof(T) * n).wait();

    for (std::size_t j = 0; j < n; j++) {
      assert(v[j] == v_serial[j]);
    }
  }

  fmt::print("oneDPL executing on one device:\n");
  fmt::print("Durations: {}\n", durations | rng::views::transform([](auto &&x) {
                                  return x * 1000;
                                }));

  std::sort(durations.begin(), durations.end());

  median_duration = durations[durations.size() / 2];

  fmt::print("Median duration: {} ms\n", median_duration * 1000);

  return 0;
}
