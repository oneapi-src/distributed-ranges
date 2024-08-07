// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cstdio>
#include <dr/sp.hpp>

int main(int argc, char **argv) {
  auto devices = dr::sp::get_numa_devices(sycl::default_selector_v);
  dr::sp::init(devices);

  dr::sp::sparse_matrix<float> x({100, 100}, 0.01);

  printf("%lu x %lu matrix with %lu stored values.\n", x.shape()[0],
         x.shape()[1], x.size());

  for (auto &&[idx, v] : x) {
    auto &&[i, j] = idx;
    printf("(%lu, %lu): %f\n", i, j, (float)v);
  }

  return 0;
}
