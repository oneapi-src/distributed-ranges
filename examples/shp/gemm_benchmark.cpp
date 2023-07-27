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

template <rng::forward_range R> auto values_view(R &&m) {
  return m | shp::views::transform([](auto &&e) {
           auto &&[_, v] = e;
           return v;
         });
}

template <typename T> auto sum_matrix(shp::distributed_dense_matrix<T> &m) {
  auto view = values_view(m);
  auto &&segments = view.segments();

  return shp::reduce(view, T(0));
}

template <typename T, typename U>
void assign(shp::distributed_dense_matrix<T> &m, const U &value) {
  dr::shp::for_each(dr::shp::par_unseq, m, [=](auto &&entry) {
    auto &&[idx, v] = entry;
    v = value;
  });
}

int main(int argc, char **argv) {
  auto devices_ = dr::shp::get_numa_devices(sycl::default_selector_v);

  std::size_t n_devices = devices_.size();

  auto devices =
      dr::shp::trim_devices(devices_, std::min(n_devices, devices_.size()));

  dr::shp::init(devices);

  fmt::print("Running with {} devices.\n", devices.size());
  dr::shp::print_device_details(devices);

  std::size_t m = 32 * 1024;
  std::size_t n = 32 * 1024;
  std::size_t k = 32 * 1024;

  auto partitions = shp::partition_matmul(m, n, k);

  using T = float;

  dr::shp::distributed_dense_matrix<T> a({m, k}, partitions[0]);
  dr::shp::distributed_dense_matrix<T> b({k, n}, partitions[1]);
  dr::shp::distributed_dense_matrix<T> c({m, n}, partitions[2]);

  dr::shp::distributed_dense_matrix<T> c_ref({m, n}, partitions[2]);

  dr::shp::distributed_dense_matrix<T> c_diff({m, n}, partitions[2]);

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

  assign(c, 0);
  assign(c_ref, 0);

  fmt::print("Warmup MatMul...\n");

  shp::gemm_inplace(a, b, c_ref);

  T single_sum = sum_matrix(c_ref);

  fmt::print("Sum: {}\n", single_sum);

  if (m <= 200 && n <= 200) {
    auto c_serial = serial_test<T>(m, n, k);

    assert(c.shape() == c_serial.shape());
    for (std::size_t i = 0; i < m; i++) {
      for (std::size_t j = 0; j < n; j++) {
        if (!is_equal<T>(c_serial[{i, j}], c_ref[{i, j}])) {
          fmt::print("{}, {}: {} != {}\n", i, j, T(c_serial[{i, j}]),
                     T(c_ref[{i, j}]));
          fmt::print("Not equal!\n");
        }
        assert(is_equal<T>(c_serial[{i, j}], c_ref[{i, j}]));
      }
    }
    delete[] c_serial.data();
  } else {
    fmt::print("Matrix too large, not performing correctness check.\n");
  }

  T total_sum(0);

  std::size_t n_iterations = 10;
  std::vector<double> durations;
  durations.reserve(n_iterations);

  for (std::size_t i = 0; i < n_iterations; i++) {
    assign(c, 0);

    auto begin = std::chrono::high_resolution_clock::now();
    shp::gemm(a, b, c);
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end - begin).count();
    durations.push_back(duration);

    T sum = sum_matrix(c);

    auto sub_view = shp::views::zip(values_view(c), values_view(c_ref)) |
                    shp::views::transform([](auto &&e) {
                      auto &&[value, ref] = e;

                      return std::abs(value - ref);
                    });

    T diff_sum = shp::reduce(shp::par_unseq, sub_view, T(0));
    fmt::print("Diff sum is {}\n", diff_sum);

    if (!is_equal<T>(diff_sum, T(0))) {
      for (std::size_t i_ = 0; i_ < c.grid_shape()[0]; i_++) {
        for (std::size_t j_ = 0; j_ < c.grid_shape()[1]; j_++) {
          auto c_tile = c.get_tile({i_, j_});
          auto cr_tile = c_ref.get_tile({i_, j_});

          bool equal = true;
          for (auto &&e :
               rng::views::zip(values_view(c_tile), values_view(cr_tile))) {
            auto &&[cv, rv] = e;
            if (cv != rv) {
              equal = false;
              break;
            }
          }
          if (!equal) {
            fmt::print("{}, {} not equal.\n", i_, j_);
            for (auto &&[ce, re] : rng::views::zip(c_tile, cr_tile)) {
              auto &&[c_idx, c_v] = ce;
              auto &&[r_idx, r_v] = re;
              if (c_v != r_v) {
                fmt::print("{} ({} vs {})\n", c_idx, c_v, r_v);
                break;
              }
            }
          }
        }
      }
    }

    // fmt::print("Max diff is {}\n", max_diff);

    /*
        fmt::print("{} vs {}\n", sum, single_sum);
        assert(is_equal<T>(sum, single_sum));
        fmt::print("Single sum is equal.\n");
        */

    total_sum += sum;
  }

  fmt::print("Average sum {}\n", total_sum / n_iterations);

  // if (n < 16 * 1024) {
  fmt::print("{} vs {}\n", total_sum / n_iterations, single_sum);
  assert(is_equal<T>(total_sum / n_iterations, single_sum));
  // }

  fmt::print("Durations: {}\n", durations);

  std::sort(durations.begin(), durations.end());

  double median_duration = durations[durations.size() / 2];

  fmt::print("Median duration: {}\n", median_duration);

  std::size_t nflops = 2 * m * n * k + 3 * m * n;

  double flop_rate = nflops / median_duration;
  double gflop_rate = flop_rate * 1e-9;
  double tflop_rate = gflop_rate * 1e-3;

  fmt::print("{} GFLOPs\n", gflop_rate);
  fmt::print("{} TFLOPs\n", tflop_rate);

  return 0;
}
