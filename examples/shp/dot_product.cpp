// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <dr/shp/shp.hpp>
#include <fmt/core.h>
#include <fmt/ranges.h>

template <dr::distributed_range X, dr::distributed_range Y>
auto dot_product_distributed(X &&x, Y &&y) {
  auto z = shp::views::zip(x, y) | dr::views::transform([](auto &&elem) {
             auto &&[a, b] = elem;
             return a * b;
           });

  return shp::reduce(shp::par_unseq, z, 0, std::plus());
}

template <rng::forward_range X, rng::forward_range Y>
auto dot_product_sequential(X &&x, Y &&y) {
  auto z = rng::views::zip(x, y) | rng::views::transform([](auto &&elem) {
             auto &&[a, b] = elem;
             return a * b;
           });

  return std::reduce(z.begin(), z.end(), 0, std::plus());
}

int main(int argc, char **argv) {
  auto devices = shp::get_numa_devices(sycl::default_selector_v);
  shp::init(devices);

  std::size_t n = 100;

  shp::distributed_vector<int> x(n);
  shp::distributed_vector<int> y(n);

  std::iota(x.begin(), x.end(), 0);
  std::iota(y.begin(), y.end(), 0);

  auto v = dot_product_distributed(x, y);

  fmt::print("{}\n", v);

  std::vector<int> x_local(n);
  std::vector<int> y_local(n);
  std::iota(x_local.begin(), x_local.end(), 0);
  std::iota(y_local.begin(), y_local.end(), 0);

  auto v_serial = dot_product_sequential(x_local, y_local);

  assert(v == v_serial);

  return 0;
}
