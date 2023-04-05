// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <dr/shp/shp.hpp>
#include <fmt/core.h>
#include <fmt/ranges.h>

#include <grb/grb.hpp>

#include <concepts>

#ifdef USE_MKL
#include <oneapi/mkl.hpp>
#endif

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
  auto devices = shp::get_numa_devices(sycl::default_selector_v);
  shp::init(devices);

  std::string fname = "/nfs/site/home/bbrock/data/com-Orkut.mtx";

  using T = float;
  using I = int;

  fmt::print("Reading in matrix file {}\n", fname);
  auto a = shp::mmread<T, I>(fname);

  auto square_block = shp::block_cyclic(
      {shp::tile::div, (a.shape()[1] + shp::nprocs() - 1) / shp::nprocs()},
      {shp::nprocs(), 1});
  auto a_square = shp::mmread<T, I>(fname, square_block);
  fmt::print("Square {} x {}\n", a_square.grid_shape()[0],
             a_square.grid_shape()[1]);

  auto c_local = local_gemv(grb::matrix<T, I>(fname));

  std::size_t m = a.shape()[0];
  std::size_t k = a.shape()[1];

  fmt::print("Initializing distributed data structures...\n");
  shp::distributed_vector<T, shp::device_allocator<T>> b(k);
  shp::distributed_vector<T, shp::device_allocator<T>> c(m);

  shp::for_each(shp::par_unseq, b, [](auto &&v) { v = 1; });
  shp::for_each(shp::par_unseq, c, [](auto &&v) { v = 0; });

  std::size_t n_iterations = 10;

  std::vector<double> durations;
  durations.reserve(n_iterations);

  fmt::print("Benchmarking...\n");

  // GEMV

  shp::gemv(c, a, b);
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

  // Square GEMV
  {

    shp::for_each(shp::par_unseq, c, [](auto &&v) { v = 0; });
    shp::gemv_square(c, a_square, b);
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

    std::cout << "Square Row-based iteration: " << median_duration * 1000
              << " ms" << std::endl;

    std::size_t n_bytes = sizeof(T) * a.size() +
                          sizeof(I) * (a.size() + a.shape()[0] + 1) // size of A
                          + sizeof(T) * b.size()                    // size of B
                          + sizeof(T) * c.size();                   // size of C
    double n_gbytes = n_bytes * 1e-9;
    fmt::print("{} GB/s\n", n_gbytes / median_duration);

    durations.clear();
  }

#ifdef USE_MKL

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

    oneapi::mkl::sparse::matrix_handle_t a_handle;
    oneapi::mkl::sparse::init_matrix_handle(&a_handle);

    oneapi::mkl::sparse::set_csr_data(
        a_handle, local_mat.shape()[0], local_mat.shape()[1],
        oneapi::mkl::index_base::zero, rowptr, colind, values);

    auto e = oneapi::mkl::sparse::gemv(q, oneapi::mkl::transpose::nontrans, 1.0,
                                       a_handle, x.data().get_raw_pointer(),
                                       0.0, y.data().get_raw_pointer());
    e.wait();

    for (std::size_t i = 0; i < n_iterations; i++) {
      auto begin = std::chrono::high_resolution_clock::now();
      auto e = oneapi::mkl::sparse::gemv(
          q, oneapi::mkl::transpose::nontrans, 1.0, a_handle,
          x.data().get_raw_pointer(), 0.0, y.data().get_raw_pointer());
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

#endif

  shp::finalize();
  return 0;
}
