// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <dr/shp/shp.hpp>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <ranges>

template <lib::distributed_range X, lib::distributed_range Y>
auto dot_product(X &&x, Y &&y) {
  auto z = shp::views::zip(x, y) | lib::views::transform([](auto &&elem) {
             auto &&[a, b] = elem;
             return a * b;
           });

  return shp::reduce(shp::par_unseq, z, 0, std::plus());
}

int main(int argc, char **argv) {
  auto devices = shp::get_numa_devices(sycl::default_selector_v);
  shp::init(devices);

  std::size_t n = 100;

  shp::distributed_vector<int> x(n);
  shp::distributed_vector<int> y(n);

  std::iota(x.begin(), x.end(), 0);
  std::iota(y.begin(), y.end(), 0);

  auto v = dot_product(x, y);

  fmt::print("{}\n", v);

  return 0;
}
