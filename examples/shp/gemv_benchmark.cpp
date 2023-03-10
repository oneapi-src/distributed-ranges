// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <dr/shp/algorithms/gemv.hpp>
#include <dr/shp/containers/sparse_matrix.hpp>
#include <dr/shp/shp.hpp>
#include <fmt/core.h>
#include <fmt/ranges.h>

template <typename Matrix> void iterate_row(Matrix &&a) {
  std::vector<sycl::event> events;
  events.reserve(a.tiles().size());

  for (auto &&tile : a.tiles()) {
    auto device = shp::devices()[tile.rank()];

    sycl::queue q(shp::context(), device);

    std::size_t wg = 32;

    sycl::event e = q.parallel_for(sycl::nd_range<1>(tile.shape()[0] * wg, wg),
                                   [=](auto &&item) {
                                     auto row_index = item.get_group(0);
                                     auto local_id = item.get_local_id();
                                     auto group_size = item.get_local_range(0);

                                     auto row = tile.row(row_index);

                                     for (std::size_t idx = local_id;
                                          idx < row.size(); idx += group_size) {
                                       auto &&[index, v] = row[idx];
                                       auto &&[i, j] = index;

                                       v = v + i + j;
                                     }
                                   });
    events.push_back(e);
  }

  for (auto &&e : events) {
    e.wait();
  }
}

template <typename Matrix> void iterate_flat(Matrix &&a) {
  std::vector<sycl::event> events;
  events.reserve(a.tiles().size());

  for (auto &&tile : a.tiles()) {
    auto device = shp::devices()[tile.rank()];

    sycl::queue q(shp::context(), device);

    auto first = tile.begin();

    sycl::event e = q.parallel_for(tile.size(), [=](auto &&id) {
      auto &&[index, v] = *(first + id);
      auto &&[i, j] = index;

      v = v + i + j;
    });
    events.push_back(e);
  }

  for (auto &&e : events) {
    e.wait();
  }
}

void hierarchical_test() {
  auto device = shp::devices()[0];

  sycl::queue q(shp::context(), device);

  std::size_t n = 100;

  int *mem = sycl::malloc_shared<int>(n * 3, device, shp::context());

  std::size_t wg = 32;

  std::size_t range_size = wg * ((n + wg - 1) / wg);

  q.parallel_for(sycl::nd_range<1>(range_size, wg), [=](auto &&item) {
    auto global_id = item.get_global_id();
    auto local_id = item.get_local_id();
    auto group_id = item.get_group(0);
    if (global_id < n) {
      mem[global_id] = global_id;
      mem[global_id + n] = local_id;
      mem[global_id + n * 2] = group_id;
    }
  });

  for (size_t i = 0; i < n; i++) {
    fmt::print("Work item {} has local_id {} in group_id {}\n", mem[i],
               mem[i + n], mem[i + 2 * n]);
  }
}

int main(int argc, char **argv) {
  auto devices = shp::get_numa_devices(sycl::default_selector_v);
  shp::init(devices);

  // hierarchical_test();

  std::size_t m = 1000;
  std::size_t k = 1000;

  shp::distributed_vector<int, shp::device_allocator<int>> b(k);

  shp::for_each(shp::par_unseq, shp::enumerate(b), [](auto &&tuple) {
    auto &&[idx, value] = tuple;
    value = 1;
  });

  shp::distributed_vector<int, shp::device_allocator<int>> c(m);

  shp::for_each(shp::par_unseq, c, [](auto &&v) { v = 0; });

  shp::sparse_matrix<int> a(
      {m, k}, 0.01,
      shp::block_cyclic({shp::tile::div, shp::tile::div}, {shp::nprocs(), 1}));

  std::size_t n_iterations = 10;

  std::vector<double> durations;
  durations.reserve(n_iterations);

  for (std::size_t i = 0; i < n_iterations; i++) {
    auto begin = std::chrono::high_resolution_clock::now();
    shp::gemv_rows(c, a, b);
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end - begin).count();
    durations.push_back(duration);
  }

  std::sort(durations.begin(), durations.end());

  double median_duration = durations[durations.size() / 2];

  std::cout << "Row-based iteration: " << median_duration * 1000 << " ms"
            << std::endl;
  fmt::print("Durations: {}\n", durations | rng::views::transform([](auto &&x) {
                                  return x * 1000;
                                }));

  durations.clear();

  for (std::size_t i = 0; i < n_iterations; i++) {
    auto begin = std::chrono::high_resolution_clock::now();
    shp::gemv(c, a, b);
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end - begin).count();
    durations.push_back(duration);
  }

  std::sort(durations.begin(), durations.end());

  median_duration = durations[durations.size() / 2];

  std::cout << "Flat iteration: " << median_duration * 1000 << " ms"
            << std::endl;
  fmt::print("Durations: {}\n", durations | rng::views::transform([](auto &&x) {
                                  return x * 1000;
                                }));

  fmt::print("Printing tile...\n");
  auto &&tile = a.tile({0, 0});
  auto &&submatrix = tile.submatrix({0, 100}, {0, 100});
  // auto&& submatrix = tile;

  auto size = rng::distance(submatrix);
  fmt::print("{} nonzeros.\n", size);

  fmt::print("Reading matrix...\n");
  auto x = shp::mmread<float, std::size_t>(
      "/nfs/site/home/bbrock/data/mouse_gene.mtx");

  fmt::print("{} x {}, {} nnz\n", x.shape()[0], x.shape()[1], x.size());

  for (size_t i = 0; i < x.grid_shape()[0]; i++) {
    for (size_t j = 0; j < x.grid_shape()[1]; j++) {
      auto &&tile = x.tile({i, j});
      fmt::print("Tile {}, {}: {}, {}\n", i, j, tile.shape(), tile.size());
      /*
      for (auto&& [index, v] : x.tile({i, j})) {
        fmt::print("{}: {}\n", index, v);
      }
      */
    }
  }

  for (auto &&[index, v] : x) {
    fmt::print("{}: {}\n", index, v);
  }

  shp::finalize();
  return 0;
}
