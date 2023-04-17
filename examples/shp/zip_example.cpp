// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <dr/shp/shp.hpp>
#include <sycl/sycl.hpp>

#include <iostream>

int main(int argc, char **argv) {
  printf("Creating NUMA devices...\n");
  // auto devices = dr::shp::get_duplicated_devices(sycl::gpu_selector_v, 8);
  auto devices = dr::shp::get_numa_devices(sycl::gpu_selector_v);
  dr::shp::init(devices);

  using DV = dr::shp::distributed_vector<int, dr::shp::device_allocator<int>>;
  DV v(100);

  DV v2(50);

  dr::shp::for_each(dr::shp::par_unseq, dr::shp::enumerate(v),
                    [](auto &&tuple) {
                      auto &&[idx, value] = tuple;
                      value = idx;
                    });

  dr::shp::for_each(dr::shp::par_unseq, v, [](auto &&value) { value += 2; });

  std::size_t sum = dr::shp::reduce(dr::shp::par_unseq, v, int(0), std::plus{});

  dr::shp::print_range(v);

  dr::shp::distributed_span dspan(v.segments());
  dr::shp::print_range(dspan);

  auto i = rng::views::iota(int32_t(0), int32_t(rng::size(v)));
  dr::shp::zip_view zip_v(i, v);

  auto segments = zip_v.segments();

  dr::shp::for_each(dr::shp::par_unseq, zip_v, [](auto &&tuple) {
    auto &&[i, v] = tuple;
    v = i;
  });

  dr::shp::zip_view zip_v2(i, v2);

  dr::shp::for_each(dr::shp::par_unseq, zip_v2, [](auto &&tuple) {
    auto &&[i, v2] = tuple;
    v2 = i;
  });

  dr::shp::zip_view view2(v, v2);

  dr::shp::print_range(v, "v");
  dr::shp::print_range(v2, "v2");

  dr::shp::print_range_details(v, "v");
  dr::shp::print_range_details(v2, "v2");

  printf("Writing to zip_view...\n");
  dr::shp::for_each(dr::shp::par_unseq, view2, [](auto &&tuple) {
    auto &&[v, v2] = tuple;
    v2 = 1;
  });

  dr::shp::print_range(v);
  dr::shp::print_range(v2);

  return 0;
}
