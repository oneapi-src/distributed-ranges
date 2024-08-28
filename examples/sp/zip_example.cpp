// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <dr/sp.hpp>

int main(int argc, char **argv) {
  printf("Creating NUMA devices...\n");
  // auto devices = dr::sp::get_duplicated_devices(sycl::gpu_selector_v, 8);
  auto devices = dr::sp::get_numa_devices(sycl::gpu_selector_v);
  dr::sp::init(devices);

  using DV = dr::sp::distributed_vector<int, dr::sp::device_allocator<int>>;
  DV v(100);

  DV v2(50);

  dr::sp::for_each(dr::sp::par_unseq, dr::sp::enumerate(v), [](auto &&tuple) {
    auto &&[idx, value] = tuple;
    value = idx;
  });

  dr::sp::for_each(dr::sp::par_unseq, v, [](auto &&value) { value += 2; });

  std::size_t sum [[maybe_unused]] =
      dr::sp::reduce(dr::sp::par_unseq, v, int(0), std::plus{});

  dr::sp::print_range(v);

  dr::sp::distributed_span dspan(v.segments());
  dr::sp::print_range(dspan);

  auto i = rng::views::iota(int32_t(0), int32_t(rng::size(v)));
  dr::sp::zip_view zip_v(i, v);

  auto segments = zip_v.segments();

  dr::sp::for_each(dr::sp::par_unseq, zip_v, [](auto &&tuple) {
    auto &&[i, v] = tuple;
    v = i;
  });

  dr::sp::zip_view zip_v2(i, v2);

  dr::sp::for_each(dr::sp::par_unseq, zip_v2, [](auto &&tuple) {
    auto &&[i, v2] = tuple;
    v2 = i;
  });

  dr::sp::zip_view view2(v, v2);

  dr::sp::print_range(v, "v");
  dr::sp::print_range(v2, "v2");

  dr::sp::print_range_details(v, "v");
  dr::sp::print_range_details(v2, "v2");

  printf("Writing to zip_view...\n");
  dr::sp::for_each(dr::sp::par_unseq, view2, [](auto &&tuple) {
    auto &&[v, v2] = tuple;
    v2 = 1;
  });

  dr::sp::print_range(v);
  dr::sp::print_range(v2);

  return 0;
}
