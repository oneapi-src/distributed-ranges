// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <dr/shp.hpp>

int main(int argc, char **argv) {
  auto devices = dr::shp::get_numa_devices(sycl::gpu_selector_v);
  dr::shp::init(devices);

  auto partition = dr::shp::block_cyclic();
  dr::shp::distributed_dense_matrix<float> x({10, 10}, partition);

  x[{2, 3}] = 12;
  x[{5, 7}] = 42;
  x[{8, 9}] = 37;

  // Execute a parallel `for_each` algorithm across all the GPUs.
  // Each entry consists of an index tuple and value.  (With a dense matrix,
  // the indices are not stored explicitly.)
  //
  // Here, we add `12` to each scalar value.
  dr::shp::for_each(dr::shp::par_unseq, x, [](auto &&entry) {
    auto &&[idx, v] = entry;
    v = v + 12;
  });

  // Print out matrix.
  for (auto iter = x.begin(); iter != x.end(); ++iter) {
    auto &&[idx, v] = *iter;
    auto &&[i, j] = idx;
    std::cout << "(" << i << ", " << j << "): " << v << std::endl;
  }

  return 0;
}
