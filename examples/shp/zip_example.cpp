// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <dr/shp/shp.hpp>
#include <sycl/sycl.hpp>

#include <iostream>

int main(int argc, char **argv) {
  printf("Creating NUMA devices...\n");
  // auto devices = shp::get_duplicated_devices(sycl::gpu_selector_v, 8);
  auto devices = shp::get_numa_devices(sycl::gpu_selector_v);
  shp::init(devices);

  using DV = shp::distributed_vector<int, shp::device_allocator<int>>;
  DV v(100);

  DV v2(50);

  shp::for_each(shp::par_unseq, shp::enumerate(v), [](auto &&tuple) {
    auto &&[idx, value] = tuple;
    value = idx;
  });

  shp::for_each(shp::par_unseq, v, [](auto &&value) { value += 2; });

  size_t sum = shp::reduce(shp::par_unseq, v, int(0), std::plus{});

  shp::print_range(v);

  shp::distributed_span dspan(v.segments());
  shp::print_range(dspan);

  auto i = rng::views::iota(int32_t(0), int32_t(rng::size(v)));
  shp::zip_view zip_v(i, v);

  auto segments = zip_v.segments();

  shp::for_each(shp::par_unseq, zip_v, [](auto &&tuple) {
    auto &&[i, v] = tuple;
    v = i;
  });

  shp::zip_view zip_v2(i, v2);

  shp::for_each(shp::par_unseq, zip_v2, [](auto &&tuple) {
    auto &&[i, v2] = tuple;
    v2 = i;
  });

  shp::zip_view view2(v, v2);

  shp::print_range(v, "v");
  shp::print_range(v2, "v2");

  shp::print_range_details(v, "v");
  shp::print_range_details(v2, "v2");

  printf("Writing to zip_view...\n");
  shp::for_each(shp::par_unseq, view2, [](auto &&tuple) {
    auto &&[v, v2] = tuple;
    v2 = 1;
  });

  shp::print_range(v);
  shp::print_range(v2);

  return 0;
}
