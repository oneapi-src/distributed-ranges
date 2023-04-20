// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <dr/shp.hpp>
#include <fmt/core.h>
#include <fmt/ranges.h>

// FIXME: what is grb.hpp? add it to cmake or remove this dependency
#include <grb/grb.hpp>

#include <concepts>

namespace shp = dr::shp;

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

template <typename T, typename U> bool is_equal(T &&x, U &&y) { return x == y; }

template <std::floating_point T>
bool is_equal(T a, T b, T epsilon = 128 * std::numeric_limits<T>::epsilon()) {
  if (a == b) {
    return true;
  }

  auto abs_th = std::numeric_limits<T>::min();

  auto diff = std::abs(a - b);

  auto norm =
      std::min((std::abs(a) + std::abs(b)), std::numeric_limits<float>::max());
  return diff < std::max(abs_th, epsilon * norm);
}

template <rng::forward_range A, rng::forward_range B>
bool is_equal(A &&a, B &&b) {
  for (auto &&[x, y] : rng::views::zip(a, b)) {
    if (!is_equal(x, y)) {
      return false;
    }
  }
  return true;
}

int main(int argc, char **argv) {
  auto devices = dr::shp::get_numa_devices(sycl::default_selector_v);
  dr::shp::init(devices);

  if (argc != 2) {
    fmt::print("usage: ./gemv_benchmark [matrix market file]\n");
    return 1;
  }

  std::string fname(argv[1]);

  using T = float;
  using I = int;

  fmt::print("Reading in matrix file {}\n", fname);
  auto a = dr::shp::mmread<T, I>(fname);

  auto square_block = dr::shp::block_cyclic(
      {dr::shp::tile::div,
       (a.shape()[1] + dr::shp::nprocs() - 1) / dr::shp::nprocs()},
      {dr::shp::nprocs(), 1});
  auto a_square = dr::shp::mmread<T, I>(fname, square_block);
  fmt::print("Square {} x {}\n", a_square.grid_shape()[0],
             a_square.grid_shape()[1]);

  auto c_local = local_gemv(grb::matrix<T, I>(fname));

  std::size_t m = a.shape()[0];
  std::size_t k = a.shape()[1];

  fmt::print("Initializing distributed data structures...\n");
  dr::shp::distributed_vector<T, dr::shp::device_allocator<T>> b(k);
  dr::shp::distributed_vector<T, dr::shp::device_allocator<T>> c(m);

  dr::shp::for_each(dr::shp::par_unseq, b, [](auto &&v) { v = 1; });
  dr::shp::for_each(dr::shp::par_unseq, c, [](auto &&v) { v = 0; });

  std::size_t n_iterations = 10;

  std::vector<double> durations;
  durations.reserve(n_iterations);

  // GEMV
  fmt::print("Verification:\n");

  fmt::print("Computing GEMV...\n");
  dr::shp::gemv(c, a, b);
  fmt::print("Copying...\n");
  std::vector<T> l(c.size());
  dr::shp::copy(c.begin(), c.end(), l.begin());
  fmt::print("Verifying...\n");
  for (std::size_t i = 0; i < l.size(); i++) {
    if (!is_equal(l[i], c_local[i])) {
      fmt::print("{} != {}\n", l[i], c_local[i]);
    }
  }
  assert(is_equal(c_local, l));

  fmt::print("Benchmarking...\n");
  for (std::size_t i = 0; i < n_iterations; i++) {
    auto begin = std::chrono::high_resolution_clock::now();
    dr::shp::gemv(c, a, b);
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end - begin).count();
    durations.push_back(duration);
  }

  fmt::print("Durations: {}\n", durations | rng::views::transform([](auto &&x) {
                                  return x * 1000;
                                }));

  std::sort(durations.begin(), durations.end());

  double median_duration = durations[durations.size() / 2];

  std::cout << "GEMV Row: " << median_duration * 1000 << " ms" << std::endl;

  std::size_t n_bytes = sizeof(T) * a.size() +
                        sizeof(I) * (a.size() + a.shape()[0] + 1) // size of A
                        + sizeof(T) * b.size()                    // size of B
                        + sizeof(T) * c.size();                   // size of C
  double n_gbytes = n_bytes * 1e-9;
  fmt::print("{} GB/s\n", n_gbytes / median_duration);

  durations.clear();

  // Square GEMV
  {
    dr::shp::for_each(shp::par_unseq, c, [](auto &&v) { v = 0; });
    shp::gemv_square(c, a_square, b);
    std::vector<T> l(c.size());
    shp::copy(c.begin(), c.end(), l.begin());
    for (std::size_t i = 0; i < l.size(); i++) {
      if (!is_equal(l[i], c_local[i])) {
        // fmt::print("{} != {}\n", l[i], c_local[i]);
      }
    }
    assert(is_equal(c_local, l));

    for (std::size_t i = 0; i < n_iterations; i++) {
      auto begin = std::chrono::high_resolution_clock::now();
      shp::gemv_square(c, a_square, b);
      auto end = std::chrono::high_resolution_clock::now();
      double duration = std::chrono::duration<double>(end - begin).count();
      durations.push_back(duration);
    }

    fmt::print("Durations: {}\n",
               durations |
                   rng::views::transform([](auto &&x) { return x * 1000; }));

    std::sort(durations.begin(), durations.end());

    double median_duration = durations[durations.size() / 2];

    std::cout << "GEMV Square: " << median_duration * 1000 << " ms"
              << std::endl;

    std::size_t n_bytes = sizeof(T) * a.size() +
                          sizeof(I) * (a.size() + a.shape()[0] + 1) // size of A
                          + sizeof(T) * b.size()                    // size of B
                          + sizeof(T) * c.size();                   // size of C
    double n_gbytes = n_bytes * 1e-9;
    fmt::print("{} GB/s\n", n_gbytes / median_duration);

    durations.clear();
  }

  // Square GEMV Copy
  {
    shp::for_each(shp::par_unseq, c, [](auto &&v) { v = 0; });
    shp::gemv_square_copy(c, a_square, b);
    std::vector<T> l(c.size());
    shp::copy(c.begin(), c.end(), l.begin());
    for (std::size_t i = 0; i < l.size(); i++) {
      if (!is_equal(l[i], c_local[i])) {
        fmt::print("{} != {}\n", l[i], c_local[i]);
      }
    }
    assert(is_equal(c_local, l));

    for (std::size_t i = 0; i < n_iterations; i++) {
      auto begin = std::chrono::high_resolution_clock::now();
      shp::gemv_square_copy(c, a_square, b);
      auto end = std::chrono::high_resolution_clock::now();
      double duration = std::chrono::duration<double>(end - begin).count();
      durations.push_back(duration);
    }

    fmt::print("Durations: {}\n",
               durations |
                   rng::views::transform([](auto &&x) { return x * 1000; }));

    std::sort(durations.begin(), durations.end());

    double median_duration = durations[durations.size() / 2];

    std::cout << "GEMV Square Copy: " << median_duration * 1000 << " ms"
              << std::endl;

    std::size_t n_bytes = sizeof(T) * a.size() +
                          sizeof(I) * (a.size() + a.shape()[0] + 1) // size of A
                          + sizeof(T) * b.size()                    // size of B
                          + sizeof(T) * c.size();                   // size of C
    double n_gbytes = n_bytes * 1e-9;
    fmt::print("{} GB/s\n", n_gbytes / median_duration);

    durations.clear();
  }

  {
    auto m = shp::__detail::mmread<T, I>(fname);
    auto shape = m.shape();
    auto nnz = m.size();

    auto local_mat =
        shp::__detail::convert_to_csr(m, shape, nnz, std::allocator<T>{});

    sycl::queue q(shp::context(), shp::devices()[0]);

    T *values = sycl::malloc_device<T>(nnz, q);
    I *colind = sycl::malloc_device<I>(nnz, q);
    I *rowptr = sycl::malloc_device<I>(local_mat.shape()[0] + 1, q);

    q.memcpy(values, local_mat.values_data(), sizeof(T) * nnz).wait();
    q.memcpy(colind, local_mat.colind_data(), sizeof(T) * nnz).wait();
    q.memcpy(rowptr, local_mat.rowptr_data(),
             sizeof(T) * (local_mat.shape()[0] + 1))
        .wait();

    shp::device_allocator<T> allocator(q);

    shp::vector<T, shp::device_allocator<T>> x(local_mat.shape()[1], 1,
                                               allocator);
    shp::vector<T, shp::device_allocator<T>> y(local_mat.shape()[1], 0,
                                               allocator);

    shp::__detail::destroy_csr_matrix_view(local_mat, std::allocator<T>{});

    shp::csr_matrix_view a_view(values, rowptr, colind, shape, nnz, 0);

    auto e = shp::__detail::local_gemv(q, a_view, x.data().get_raw_pointer(),
                                       y.data().get_raw_pointer());
    e.wait();

    for (std::size_t i = 0; i < n_iterations; i++) {
      auto begin = std::chrono::high_resolution_clock::now();
      auto e = shp::__detail::local_gemv(q, a_view, x.data().get_raw_pointer(),
                                         y.data().get_raw_pointer());
      e.wait();
      auto end = std::chrono::high_resolution_clock::now();
      double duration = std::chrono::duration<double>(end - begin).count();
      durations.push_back(duration);
    }

    fmt::print("Durations: {}\n",
               durations |
                   rng::views::transform([](auto &&x) { return x * 1000; }));

    std::sort(durations.begin(), durations.end());

    double median_duration = durations[durations.size() / 2];

    std::cout << "Single GPU: " << median_duration * 1000 << " ms" << std::endl;

    std::size_t n_bytes = sizeof(T) * a.size() +
                          sizeof(I) * (a.size() + a.shape()[0] + 1) // size of A
                          + sizeof(T) * b.size()                    // size of B
                          + sizeof(T) * c.size();                   // size of C
    double n_gbytes = n_bytes * 1e-9;
    fmt::print("{} GB/s\n", n_gbytes / median_duration);

    durations.clear();
  }

  shp::finalize();
  return 0;
}
