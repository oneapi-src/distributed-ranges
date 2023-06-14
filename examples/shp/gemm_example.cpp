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
    v = static_cast<float>(idx[0] + idx[1]) / a.shape()[0];
  }

  for (auto &&[idx, v] : b) {
    v = static_cast<float>(idx[0] + idx[1]) / b.shape()[0];
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
  auto devices = dr::shp::get_duplicated_devices(sycl::default_selector_v, 16);
  dr::shp::init(devices);

  std::size_t m = 100;
  std::size_t n = 100;
  std::size_t k = 100;

  auto partitions = shp::partition_matmul(m, n, k);

  dr::shp::dense_matrix<float> a({m, k}, partitions[0]);
  dr::shp::dense_matrix<float> b({k, n}, partitions[1]);
  dr::shp::dense_matrix<float> c({m, n}, partitions[2]);

  auto shape = a.shape();
  dr::shp::for_each(dr::shp::par_unseq, a, [=](auto &&entry) {
    auto &&[idx, v] = entry;
    v = static_cast<float>(idx[0] + idx[1]) / shape[0];
  });

  shape = b.shape();
  dr::shp::for_each(dr::shp::par_unseq, b, [=](auto &&entry) {
    auto &&[idx, v] = entry;
    v = static_cast<float>(idx[0] + idx[1]) / shape[0];
  });

  shape = c.shape();
  dr::shp::for_each(dr::shp::par_unseq, c, [=](auto &&entry) {
    auto &&[idx, v] = entry;
    v = 0;
  });

  shp::gemm(a, b, c);

  auto c_serial = serial_test<float>(m, n, k);

  assert(c.shape() == c_serial.shape());

  for (std::size_t i = 0; i < m; i++) {
    for (std::size_t j = 0; j < n; j++) {
      if (!is_equal<float>(c_serial[{i, j}], c[{i, j}])) {
        fmt::print("{}, {}: {} != {}\n", i, j, c_serial[{i, j}], c[{i, j}]);
      }
      assert(is_equal<float>(c_serial[{i, j}], c[{i, j}]));
    }
  }

  delete[] c_serial.data();

  return 0;
}
