// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <dr/shp/shp.hpp>
#include <fmt/core.h>
#include <fmt/ranges.h>

#include <grb/grb.hpp>

template <grb::MatrixRange M> auto local_gemv(M &&a) {
  using T = grb::matrix_scalar_t<M>;
  std::vector<T> b(a.shape()[1], 1);
  std::vector<T> c(a.shape()[0], 0);

  for (auto &&[index, v] : a) {
    auto &&[i, k] = index;

    c[i] += v * b[k];
  }

  return c;
}

int main(int argc, char **argv) {
  auto devices = shp::get_numa_devices(sycl::default_selector_v);
  shp::init(devices);

  std::string fname = "/nfs/site/home/bbrock/data/mouse_gene.mtx";

  using T = float;
  using I = int;

  auto a = shp::mmread<T, I>(fname);

  auto c_local = local_gemv(grb::matrix<T, I>(fname));

  std::size_t m = a.shape()[0];
  std::size_t k = a.shape()[1];

  shp::distributed_vector<T, shp::device_allocator<T>> b(k);
  shp::distributed_vector<T, shp::device_allocator<T>> c(m);

  shp::for_each(shp::par_unseq, b, [](auto &&v) { v = 1; });
  shp::for_each(shp::par_unseq, c, [](auto &&v) { v = 0; });

  std::size_t n_iterations = 10;

  std::vector<double> durations;
  durations.reserve(n_iterations);

  shp::gemv(c, a, b);
  std::vector<T> l(c.size());
  shp::copy(c.begin(), c.end(), l.begin());
  for (std::size_t i = 0; i < l.size(); i++) {
    if (l[i] != c_local[i]) {
      fmt::print("{} != {}\n", l[i], c_local[i]);
    }
  }
  assert(rng::equal(c_local, l));

  for (std::size_t i = 0; i < n_iterations; i++) {
    auto begin = std::chrono::high_resolution_clock::now();
    shp::gemv(c, a, b);
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end - begin).count();
    durations.push_back(duration);
  }

  fmt::print("Durations: {}\n", durations | rng::views::transform([](auto &&x) {
                                  return x * 1000;
                                }));

  std::sort(durations.begin(), durations.end());

  double median_duration = durations[durations.size() / 2];

  std::cout << "Row-based iteration: " << median_duration * 1000 << " ms"
            << std::endl;

  std::size_t n_bytes = sizeof(T) * a.size() +
                        sizeof(I) * (a.size() + a.shape()[0] + 1) // size of A
                        + sizeof(T) * b.size()                    // size of B
                        + sizeof(T) * c.size();                   // size of C
  double n_gbytes = n_bytes * 1e-9;
  fmt::print("{} GB/s\n", n_gbytes / median_duration);

  durations.clear();

  shp::for_each(shp::par_unseq, c, [](auto &&v) { v = 0; });

  shp::gemv(c, a, b);
  shp::copy(c.begin(), c.end(), l.begin());
  assert(rng::equal(c_local, l));

  for (std::size_t i = 0; i < n_iterations; i++) {
    auto begin = std::chrono::high_resolution_clock::now();
    shp::flat_gemv(c, a, b);
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end - begin).count();
    durations.push_back(duration);
  }

  fmt::print("Durations: {}\n", durations | rng::views::transform([](auto &&x) {
                                  return x * 1000;
                                }));

  std::sort(durations.begin(), durations.end());

  median_duration = durations[durations.size() / 2];

  fmt::print("{} GB/s\n", n_gbytes / median_duration);

  std::cout << "Flat iteration: " << median_duration * 1000 << " ms"
            << std::endl;

  shp::finalize();
  return 0;
}
