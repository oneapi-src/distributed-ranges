// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cstdio>
#include <dr/shp/shp.hpp>

int main(int argc, char **argv) {
  auto devices = shp::get_numa_devices(sycl::gpu_selector_v);
  shp::init(devices);

  shp::sparse_matrix<float> x({100, 100}, 0.01);

  printf("%lu x %lu matrix with %lu stored values.\n", x.shape()[0],
         x.shape()[1], x.size());

  /*
    shp::for_each(shp::par_unseq, x,
                  [](auto&& entry) {
                    auto&& [idx, v] = entry;
                    auto&& [i, j] = idx;
                    v = i*1000 + j;
                  });

    std::size_t count = 0;
    for (auto&& tile : x.segments()) {
      printf("Tile %lu (%lu x %lu)\n", count++, tile.shape()[0],
    tile.shape()[1]);

      for (auto&& [idx, v] : tile) {
        auto&& [i, j] = idx;
        printf("%lu, %lu: %f\n", i, j, (float) v);
      }
    }
    */

  for (auto &&[idx, v] : x) {
    auto &&[i, j] = idx;
    printf("(%lu, %lu): %f\n", i, j, (float)v);
  }

  return 0;
}
