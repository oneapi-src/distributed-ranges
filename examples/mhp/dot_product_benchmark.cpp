// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <random>

#include <sycl/sycl.hpp>

#include "cxxopts.hpp"

#include <dr/mhp.hpp>

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/async>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/numeric>

namespace mhp = dr::mhp;

using T = float;

MPI_Comm comm;
std::size_t comm_rank;
std::size_t comm_size;

std::size_t n;
std::size_t n_iterations;

T tolerance = .0001;

bool is_equal(auto a, auto b, auto epsilon) { return a == b; }

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

auto dot_product_distributed(dr::distributed_range auto &&x,
                             dr::distributed_range auto &&y) {
  auto z = mhp::views::zip(x, y) | mhp::views::transform([](auto &&elem) {
             auto &&[a, b] = elem;
             return a * b;
           });

  return mhp::reduce(z);
}

template <rng::forward_range X, rng::forward_range Y>
auto dot_product_onedpl(sycl::queue q, X &&x, Y &&y) {
  auto z = rng::views::zip(x, y) | mhp::views::transform([](auto &&elem) {
             auto &&[a, b] = elem;
             return a * b;
           });
  oneapi::dpl::execution::device_policy policy(q);
  dr::__detail::direct_iterator d_first(z.begin());
  dr::__detail::direct_iterator d_last(z.end());
  return oneapi::dpl::experimental::reduce_async(
             policy, d_first, d_last, rng::range_value_t<X>(0), std::plus())
      .get();
}

template <rng::forward_range X, rng::forward_range Y>
auto dot_product_sequential(X &&x, Y &&y) {
  auto z = rng::views::zip(x, y) | rng::views::transform([](auto &&elem) {
             auto &&[a, b] = elem;
             return a * b;
           });

  return std::reduce(z.begin(), z.end(), rng::range_value_t<X>(0), std::plus());
}

void stats(auto &durations, auto &sum, auto v_serial, auto &x_local,
           auto &y_local) {
  fmt::print("MHP executing on {} ranks:\n", comm_size);
  fmt::print("Durations: {}\n", durations | rng::views::transform([](auto &&x) {
                                  return x * 1000;
                                }));

  std::sort(durations.begin(), durations.end());

  double median_duration = durations[durations.size() / 2];

  fmt::print("Median duration: {} ms\n", median_duration * 1000);
  fmt::print("Result: {}\n", sum);

  // Execute on one device:
  durations.clear();
  durations.reserve(n_iterations);

  sycl::queue q;

  T *x_d = sycl::malloc_device<T>(n, q);
  T *y_d = sycl::malloc_device<T>(n, q);
  q.memcpy(x_d, x_local.data(), n * sizeof(T)).wait();
  q.memcpy(y_d, y_local.data(), n * sizeof(T)).wait();

  sum = 0;
  for (std::size_t i = 0; i < n_iterations; i++) {
    std::span<T> x(x_d, n);
    std::span<T> y(y_d, n);
    auto begin = std::chrono::high_resolution_clock::now();
    auto v = dot_product_onedpl(q, x, y);
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end - begin).count();
    durations.push_back(duration);
    if (!is_equal(v, v_serial, tolerance)) {
      fmt::print("DPL dot product result mismatch:\n"
                 "  expected: {}\n"
                 "  actual:   {}\n",
                 v_serial, v);
      exit(1);
    }
    sum += v;
  }

  fmt::print("oneDPL executing on one device:\n");
  fmt::print("Durations: {}\n", durations | rng::views::transform([](auto &&x) {
                                  return x * 1000;
                                }));

  std::sort(durations.begin(), durations.end());

  median_duration = durations[durations.size() / 2];

  fmt::print("Median duration: {} ms\n", median_duration * 1000);

  fmt::print("Result: {}\n", sum);
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  comm = MPI_COMM_WORLD;
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  comm_rank = rank;
  comm_size = size;

  mhp::init();

  cxxopts::Options options_spec(argv[0], "mhp dot product benchmark");
  // clang-format off
  options_spec.add_options()
    ("n", "Size of array", cxxopts::value<std::size_t>()->default_value("1000000"))
    ("i", "Number of iterations", cxxopts::value<std::size_t>()->default_value("10"))
    ("help", "Print help");
  // clang-format on

  cxxopts::ParseResult options;
  try {
    options = options_spec.parse(argc, argv);
  } catch (const cxxopts::OptionParseException &e) {
    std::cout << options_spec.help() << "\n";
    exit(1);
  }

  n = options["n"].as<std::size_t>();
  n_iterations = options["i"].as<std::size_t>();
  std::vector<T> x_local(n);
  std::vector<T> y_local(n);
#if 0
  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_real_distribution<> dist(0, 10);
  for (std::size_t i = 0; i < n; i++) {
    x_local[i] = dist(rng);
    y_local[i] = dist(rng);
  }
#else
  rng::iota(x_local, 0);
  rng::iota(y_local, 0);
#endif

  auto v_serial = dot_product_sequential(x_local, y_local);

  mhp::distributed_vector<T> x(n);
  mhp::distributed_vector<T> y(n);
#if 0
  rng::copy(x_local, x.begin());
  rng::copy(y_local, y.begin());
#else
  mhp::iota(x, 0);
  mhp::iota(y, 0);
#endif

  fmt::print("local: {}\n", rng::views::drop(x_local, n - 10));
  fmt::print("dist: {}\n", rng::views::drop(x, n - 10));
  std::vector<double> durations;
  durations.reserve(n_iterations);

  // Execute on all devices with MHP:
  T sum = 0;
  int error = 0;
  for (std::size_t i = 0; i < n_iterations; i++) {
    auto begin = std::chrono::high_resolution_clock::now();
    auto v = dot_product_distributed(x, y);
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end - begin).count();
    durations.push_back(duration);
    if (comm_rank == 0) {
      if (!is_equal(v, v_serial, tolerance)) {
        fmt::print("DR dot product result mismatch:\n"
                   "  expected: {}\n"
                   "  actual:   {}\n",
                   v_serial, v);
        error = 1;
      }
    }
    sum += v;
  }

  if (comm_rank == 0) {
    stats(durations, sum, v_serial, x_local, y_local);
  }

  return error;
}
