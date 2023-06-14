// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <dr/shp.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>

namespace shp = dr::shp;

template <typename T>
auto serial_test(std::size_t m, std::size_t n, std::size_t k) {
  T *a_d = new T[m * k * sizeof(T)];
  T *b_d = new T[k * n * sizeof(T)];
  T *c_d = new T[m * n * sizeof(T)];

  shp::dense_matrix_view<T> a(a_d, {m, k}, k, 0);
  shp::dense_matrix_view<T> b(b_d, {k, n}, n, 0);
  shp::dense_matrix_view<T> c(c_d, {m, n}, n, 0);

  for (auto &&[idx, v] : a) {
    v = static_cast<T>(idx[0] + idx[1]) / a.shape()[0];
  }

  for (auto &&[idx, v] : b) {
    v = static_cast<T>(idx[0] + idx[1]) / b.shape()[0];
  }

  for (auto &&[idx, v] : c) {
    v = 0;
  }

  for (std::size_t i = 0; i < m; i++) {
    for (std::size_t k_ = 0; k_ < k; k_++) {
      for (std::size_t j = 0; j < n; j++) {
        c[{i, j}] += a[{i, k_}] * b[{k_, j}];
      }
    }
  }

  delete[] a_d;
  delete[] b_d;

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
      std::min((std::abs(a) + std::abs(b)), std::numeric_limits<T>::max());
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

template <typename T> auto sum_matrix(shp::dense_matrix<T> &m) {
  auto view = m | shp::views::transform([](auto &&e) {
                auto &&[_, v] = e;
                return v;
              });

  auto &&segments = view.segments();

  return shp::reduce(view, T(0));
}

int main(int argc, char **argv) {
  auto devices = dr::shp::get_duplicated_devices(sycl::default_selector_v, 16);
  dr::shp::init(devices);

  std::size_t m = 500;
  std::size_t n = 500;
  std::size_t k = 500;

  auto partitions = shp::partition_matmul(m, n, k);

  using T = float;

  dr::shp::dense_matrix<T> a({m, k}, partitions[0]);
  dr::shp::dense_matrix<T> b({k, n}, partitions[1]);
  dr::shp::dense_matrix<T> c({m, n}, partitions[2]);

  auto shape = a.shape();
  dr::shp::for_each(dr::shp::par_unseq, a, [=](auto &&entry) {
    auto &&[idx, v] = entry;
    v = static_cast<T>(idx[0] + idx[1]) / shape[0];
  });

  shape = b.shape();
  dr::shp::for_each(dr::shp::par_unseq, b, [=](auto &&entry) {
    auto &&[idx, v] = entry;
    v = static_cast<T>(idx[0] + idx[1]) / shape[0];
  });

  shape = c.shape();
  dr::shp::for_each(dr::shp::par_unseq, c, [=](auto &&entry) {
    auto &&[idx, v] = entry;
    v = 0;
  });

  fmt::print("Warmup MatMul...\n");

  shp::gemm(a, b, c);

  T single_sum = sum_matrix(c);

  fmt::print("Sum: {}\n", single_sum);

  auto c_serial = serial_test<T>(m, n, k);

  assert(c.shape() == c_serial.shape());

  for (std::size_t i = 0; i < m; i++) {
    for (std::size_t j = 0; j < n; j++) {
      if (!is_equal<T>(c_serial[{i, j}], c[{i, j}])) {
        // fmt::print("{}, {}: {} != {}\n", i, j, c_serial[{i, j}], c[{i, j}]);
        fmt::print("Not equal!\n");
      }
      assert(is_equal<T>(c_serial[{i, j}], c[{i, j}]));
    }
  }

  delete[] c_serial.data();

  T total_sum(0);

  std::size_t n_iterations = 10;
  std::vector<double> durations;
  durations.reserve(n_iterations);

  for (std::size_t i = 0; i < n_iterations; i++) {
    dr::shp::for_each(dr::shp::par_unseq, c, [=](auto &&entry) {
      auto &&[idx, v] = entry;
      v = 0;
    });

    auto begin = std::chrono::high_resolution_clock::now();
    shp::gemm(a, b, c);
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end - begin).count();
    durations.push_back(duration);

    total_sum += sum_matrix(c);
  }

  fmt::print("Average sum {}\n", total_sum / n_iterations);

  assert(is_equal<T>(total_sum / n_iterations, single_sum));

  fmt::print("Durations: {}\n", durations);

  std::sort(durations.begin(), durations.end());

  double median_duration = durations[durations.size() / 2];

  fmt::print("Median duration: {}\n", median_duration);

  std::size_t nflops = 3 * m * n * k + 2 * m * n;

  double flop_rate = nflops / median_duration;
  double gflop_rate = flop_rate * 1e-9;

  fmt::print("{} GFLOPs\n", gflop_rate);

  return 0;
}
