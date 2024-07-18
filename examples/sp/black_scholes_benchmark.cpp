// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cmath>
#include <concepts>
#include <dr/sp.hpp>
#include <vector>

#include <fmt/core.h>
#include <fmt/ranges.h>

namespace sp = dr::sp;

template <typename T> static T RandFloat(T a, T b) {
  return drand48() / std::numeric_limits<T>::max() * (b - a) + a;
}

template <typename T> auto InitData(std::size_t nopt) {
  std::vector<T> s0(nopt);
  std::vector<T> x(nopt);
  std::vector<T> t(nopt);
  std::vector<T> vcall(nopt);
  std::vector<T> vput(nopt);
  int i;

  constexpr T S0L = 10;
  constexpr T S0H = 50;
  constexpr T XL = 10;
  constexpr T XH = 50;
  constexpr T TL = 1;
  constexpr T TH = 2;

/* NUMA-friendly data init */
#pragma omp parallel for ordered
  for (i = 0; i < nopt; i++) {
#pragma omp ordered
    {
      s0[i] = RandFloat(S0L, S0H);
      x[i] = RandFloat(XL, XH);
      t[i] = RandFloat(TL, TH);
    }

    vcall[i] = 0;
    vput[i] = 0;
  }
  return std::tuple(s0, x, t, vcall, vput);
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

template <rng::range R> void fill_random(R &&r) {
  for (auto &&v : r) {
    v = drand48();
  }
}

template <std::floating_point T> T normalCDF(T x) {
  return std::erfc(-x / std::sqrt(2)) / 2;
}

template <typename T, rng::range RS, rng::range RX, rng::range RT,
          rng::range RC, rng::range RP>
void black_scholes(T r, T sig, RS &&s0, RX &&x, RT &&t, RC &&vcall, RP &&vput) {

  std::size_t nopt = rng::size(s0);

  for (std::size_t i = 0; i < nopt; i++) {
    T d1 = (std::log(s0[i] / x[i]) + (r + T(0.5) * sig * sig) * t[i]) /
           (sig * std::sqrt(t[i]));
    T d2 = (std::log(s0[i] / x[i]) + (r - T(0.5) * sig * sig) * t[i]) /
           (sig * std::sqrt(t[i]));

    vcall[i] =
        s0[i] * normalCDF(d1) - std::exp(-r * t[i]) * x[i] * normalCDF(d2);
    vput[i] =
        std::exp(-r * t[i]) * x[i] * normalCDF(-d2) - s0[i] * normalCDF(-d1);
  }
}

template <typename T, rng::range RS, rng::range RX, rng::range RT,
          rng::range RC, rng::range RP>
void black_scholes_functional(T r, T sig, RS &&s0, RX &&x, RT &&t, RC &&vcall,
                              RP &&vput) {

  auto black_scholes = [=](auto &&tuple) {
    auto &&[s0, x, t, vcall, vput] = tuple;
    T a = std::log(s0 / x);
    T b = t * -r;
    T z = t * sig * sig * 2;

    T c = T(0.25) * z;
    T e = std::exp(b);
    T y = 1 / std::sqrt(z);

    T w1 = (a - b + c) * y;
    T w2 = (a - b - c) * y;
    T d1 = std::erf(w1);
    T d2 = std::erf(w2);
    d1 = T(0.5) + T(0.5) * d1;
    d2 = T(0.5) + T(0.5) * d2;

    vcall = s0 * d1 - x * e * d2;
    vput = vcall - s0 + x * e;
  };

  auto zipped_view = rng::views::zip(s0, x, t, vcall, vput);

  std::for_each(zipped_view.begin(), zipped_view.end(), black_scholes);
}

template <typename Policy, typename T, rng::range RS, rng::range RX,
          rng::range RT, rng::range RC, rng::range RP>
void black_scholes_onedpl(Policy &&policy, T r, T sig, RS &&s0, RX &&x, RT &&t,
                          RC &&vcall, RP &&vput) {

  auto black_scholes = [=](auto &&tuple) {
    auto &&[s0, x, t, vcall, vput] = tuple;
    T a = std::log(s0 / x);
    T b = t * -r;
    T z = t * sig * sig * 2;

    T c = T(0.25) * z;
    T e = std::exp(b);
    T y = 1 / std::sqrt(z);

    T w1 = (a - b + c) * y;
    T w2 = (a - b - c) * y;
    T d1 = std::erf(w1);
    T d2 = std::erf(w2);
    d1 = T(0.5) + T(0.5) * d1;
    d2 = T(0.5) + T(0.5) * d2;

    vcall = s0 * d1 - x * e * d2;
    vput = vcall - s0 + x * e;
  };

  auto zipped_view = rng::views::zip(s0, x, t, vcall, vput);

  dr::__detail::direct_iterator d_first(zipped_view.begin());
  dr::__detail::direct_iterator d_last(zipped_view.end());

  oneapi::dpl::experimental::for_each_async(policy, d_first, d_last,
                                            black_scholes)
      .wait();
}

template <dr::distributed_iterator Iter> void iter(Iter it) {}

template <dr::distributed_range R> void range(R &&) {}

template <typename T, dr::distributed_range RS, dr::distributed_range RX,
          dr::distributed_range RT, dr::distributed_range RC,
          dr::distributed_range RP>
void black_scholes_distributed(T r, T sig, RS &&s0, RX &&x, RT &&t, RC &&vcall,
                               RP &&vput) {

  auto black_scholes = [=](auto &&tuple) {
    auto &&[s0, x, t, vcall, vput] = tuple;
    T a = std::log(s0 / x);
    T b = t * -r;
    T z = t * sig * sig * 2;

    T c = T(0.25) * z;
    T e = std::exp(b);
    T y = 1 / std::sqrt(z);

    T w1 = (a - b + c) * y;
    T w2 = (a - b - c) * y;
    T d1 = std::erf(w1);
    T d2 = std::erf(w2);
    d1 = T(0.5) + T(0.5) * d1;
    d2 = T(0.5) + T(0.5) * d2;

    vcall = s0 * d1 - x * e * d2;
    vput = vcall - s0 + x * e;
  };

  auto zipped_view = sp::views::zip(s0, x, t, vcall, vput);

  sp::for_each(zipped_view, black_scholes);
}

int main(int argc, char **argv) {

  if (argc != 3) {
    fmt::print("usage: ./dot_product_benchmark [n_devices] [n_elements]\n");
    return 1;
  }

  std::size_t n_devices = std::atoll(argv[1]);

  std::size_t n = std::atoll(argv[2]);

  // auto devices_ = dr::sp::get_numa_devices(sycl::default_selector_v);
  auto devices_ = dr::sp::get_devices(sycl::default_selector_v);

  // std::size_t n_devices = devices_.size();

  auto devices =
      dr::sp::trim_devices(devices_, std::min(n_devices, devices_.size()));

  dr::sp::init(devices);

  fmt::print("Running with {} devices, {} elements\n", devices.size(), n);
  dr::sp::print_device_details(devices);

  // std::size_t n = 2ll * 1000 * 1000 * 1000;
  using T = float;

  // Risk-free rate
  T r = 0.1;
  // Volatility
  T sig = 0.2;

  auto &&[s0, x, t, vcall, vput] = InitData<T>(n);

  std::size_t n_iterations = 10;
  T sum = 0;

  std::vector<double> durations;
  durations.reserve(n_iterations);

  {
    sp::distributed_vector<T> d_s0(n);
    sp::distributed_vector<T> d_x(n);

    sp::distributed_vector<T> d_t(n);
    sp::distributed_vector<T> d_vcall(n);
    sp::distributed_vector<T> d_vput(n);

    sp::copy(s0.begin(), s0.end(), d_s0.begin());
    sp::copy(x.begin(), x.end(), d_x.begin());
    sp::copy(t.begin(), t.end(), d_t.begin());
    sp::copy(vcall.begin(), vcall.end(), d_vcall.begin());
    sp::copy(vput.begin(), vput.end(), d_vput.begin());

    black_scholes_distributed(r, sig, d_s0, d_x, d_t, d_vcall, d_vput);

    black_scholes_functional(r, sig, s0, x, t, vcall, vput);

    fmt::print("Executing distributed...\n");
    for (std::size_t i = 0; i < n_iterations; i++) {
      auto begin = std::chrono::high_resolution_clock::now();
      black_scholes_distributed(r, sig, d_s0, d_x, d_t, d_vcall, d_vput);
      auto end = std::chrono::high_resolution_clock::now();
      double duration = std::chrono::duration<double>(end - begin).count();

      sum += sp::reduce(d_vcall);
      sum += sp::reduce(d_vput);
      durations.push_back(duration);
    }

    fmt::print("Sum: {}\n", sum);

    fmt::print("Durations: {}\n", durations);

    std::sort(durations.begin(), durations.end());

    double median_duration = durations[durations.size() / 2];

    std::cout << "Distributed Black-Scholes: " << median_duration * 1000
              << " ms" << std::endl;

    durations.clear();
  }

  if (n < 1 * 1000 * 1000 * 1000) {
    fmt::print("Executing serial Black-Scholes...\n");

    sum = 0;
    for (std::size_t i = 0; i < n_iterations; i++) {
      auto begin = std::chrono::high_resolution_clock::now();
      black_scholes(r, sig, s0, x, t, vcall, vput);
      auto end = std::chrono::high_resolution_clock::now();
      double duration = std::chrono::duration<double>(end - begin).count();

      sum += std::reduce(vcall.begin(), vcall.end());
      sum += std::reduce(vput.begin(), vput.end());
      durations.push_back(duration);
    }

    fmt::print("Sum: {}\n", sum);

    fmt::print("Durations: {}\n", durations);

    std::sort(durations.begin(), durations.end());

    double median_duration = durations[durations.size() / 2];

    std::cout << "Single-threaded Black-Scholes: " << median_duration * 1000
              << " ms" << std::endl;

    durations.clear();
  } else {
    fmt::print("n > 1*1000*1000*1000, not running serial Black-Scholes\n");
  }

  {
    auto &&q = sp::__detail::queue(0);
    oneapi::dpl::execution::device_policy policy(q);

    fmt::print("Allocating and copying over to device...\n");
    T *p_s0 = sycl::malloc_device<T>(n, q);
    T *p_x = sycl::malloc_device<T>(n, q);
    T *p_t = sycl::malloc_device<T>(n, q);
    T *p_vcall = sycl::malloc_device<T>(n, q);
    T *p_vput = sycl::malloc_device<T>(n, q);

    std::span<T> d_s0(p_s0, n);
    std::span<T> d_x(p_x, n);
    std::span<T> d_t(p_t, n);
    std::span<T> d_vcall(p_vcall, n);
    std::span<T> d_vput(p_vput, n);

    sp::copy(s0.begin(), s0.end(), p_s0);
    sp::copy(x.begin(), x.end(), p_x);
    sp::copy(t.begin(), t.end(), p_t);
    sp::copy(vcall.begin(), vcall.end(), p_vcall);
    sp::copy(vput.begin(), vput.end(), p_vput);

    fmt::print("Running...\n");
    sum = 0;
    for (std::size_t i = 0; i < n_iterations; i++) {
      auto begin = std::chrono::high_resolution_clock::now();
      black_scholes_onedpl(policy, r, sig, d_s0, d_x, d_t, d_vcall, d_vput);
      auto end = std::chrono::high_resolution_clock::now();
      double duration = std::chrono::duration<double>(end - begin).count();

      sum += oneapi::dpl::reduce(policy, p_vcall, p_vcall + n);
      sum += oneapi::dpl::reduce(policy, p_vput, p_vput + n);
      durations.push_back(duration);
    }

    fmt::print("Sum: {}\n", sum);

    fmt::print("Durations: {}\n", durations);

    std::sort(durations.begin(), durations.end());

    double median_duration = durations[durations.size() / 2];

    std::cout << "oneDPL Black-Scholes: " << median_duration * 1000 << " ms"
              << std::endl;

    durations.clear();
  }

  return 0;
}
