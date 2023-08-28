// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <dr/shp.hpp>
#include <sycl/sycl.hpp>

#include <iostream>

#include <fmt/core.h>
#include <fmt/ranges.h>

template <rng::forward_range V, rng::forward_range O>
auto inclusive_scan_onedpl(sycl::queue q, V &&v, O &&o) {
  oneapi::dpl::execution::device_policy policy(q);
  dr::__detail::direct_iterator d_first(v.begin());
  dr::__detail::direct_iterator d_last(v.end());
  dr::__detail::direct_iterator d_d_first(o.begin());
  return oneapi::dpl::inclusive_scan(policy, d_first, d_last, d_d_first,
                                     std::plus<>{});
}

int main(int argc, char **argv) {
  if (argc != 3) {
    fmt::print("usage: ./inclusive_scan_benchmark [n_devices] [n_elements]\n");
    return 1;
  }

  std::size_t n_devices = std::atoll(argv[1]);

  std::size_t n = std::atoll(argv[2]);

  auto devices_ = dr::shp::get_numa_devices(sycl::default_selector_v);

  // std::size_t n_devices = devices_.size();

  auto devices =
      dr::shp::trim_devices(devices_, std::min(n_devices, devices_.size()));

  dr::shp::init(devices);

  using T = float;

  fmt::print("Running with {} devices, {} elements.\n", devices.size(), n);
  dr::shp::print_device_details(devices);

  dr::shp::distributed_vector<T, dr::shp::device_allocator<T>> v(n);
  dr::shp::distributed_vector<T, dr::shp::device_allocator<T>> o(n);

  std::vector<T> lv(n);
  std::vector<T> ov(n);

  for (std::size_t i = 0; i < lv.size(); i++) {
    lv[i] = drand48();
  }

  std::inclusive_scan(lv.begin(), lv.end(), ov.begin());

  std::size_t n_iterations = 10;

  std::vector<double> durations;
  durations.reserve(n_iterations);

  T sum = 0;
  for (std::size_t i = 0; i < n_iterations; i++) {
    dr::shp::copy(lv.begin(), lv.end(), v.begin());

    auto begin = std::chrono::high_resolution_clock::now();
    dr::shp::inclusive_scan(dr::shp::par_unseq, v, o);
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end - begin).count();

    durations.push_back(duration);

    sum += dr::shp::reduce(o);
  }

  fmt::print("SHP executing on {} devices:\n", devices.size());
  fmt::print("Durations: {}\n", durations | rng::views::transform([](auto &&x) {
                                  return x * 1000;
                                }));

  std::sort(durations.begin(), durations.end());

  double median_duration = durations[durations.size() / 2];

  fmt::print("Median duration: {} ms\n", median_duration * 1000);

  fmt::print("Result: {}\n", sum);

  durations.clear();

  // Execution on one device with oneDPL

  sycl::queue q(dr::shp::context(), dr::shp::devices()[0]);

  T *v_d = sycl::malloc_device<T>(n, q);
  T *o_d = sycl::malloc_device<T>(n, q);
  q.memcpy(v_d, lv.data(), n * sizeof(T)).wait();

  sum = 0;
  for (std::size_t i = 0; i < n_iterations; i++) {
    std::span<T> v(v_d, n);
    std::span<T> o(o_d, n);
    auto begin = std::chrono::high_resolution_clock::now();
    inclusive_scan_onedpl(q, v, o);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double>(end - begin).count();
    durations.push_back(duration);

    oneapi::dpl::execution::device_policy policy(q);
    sum += oneapi::dpl::reduce(policy, dr::__detail::direct_iterator(o.begin()),
                               dr::__detail::direct_iterator(o.end()));
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
