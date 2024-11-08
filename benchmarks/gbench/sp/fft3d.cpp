// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "cxxopts.hpp"
#include "oneapi/mkl/dft.hpp"
#include <complex>
#include <dr/sp.hpp>
#include <fmt/core.h>
#include <latch>
#include <thread>

#ifndef STANDALONE_BENCHMARK

#include "../common/dr_bench.hpp"

#endif

template <rng::forward_range R> auto values_view(R &&m) {
  return m | dr::sp::views::transform([](auto &&e) {
           auto &&[_, v] = e;
           return v;
         });
}

using real_t = float;
using value_t = std::complex<real_t>;

template <typename T>
void init_matrix_3d(dr::sp::distributed_dense_matrix<std::complex<T>> &m,
                    int N) {
  const int N1 = N;
  const int N2 = m.shape()[1] / N;
  const int N3 = m.shape()[0];
  const int H1 = -1;
  const int H2 = -2;
  const int H3 = -3;
  constexpr T TWOPI = 6.2831853071795864769;

  auto moda = [](int K, int L, int M) { return (T)(((long long)K * L) % M); };

  const T norm = T(1) / (N3 * N2 * N1);
  dr::sp::for_each(dr::sp::par_unseq, m, [=](auto &&entry) {
    auto &&[idx, v] = entry;
    int n3 = idx[0];
    int n2 = idx[1] / N1;
    int n1 = idx[1] % N1;
    T phase = TWOPI * (moda(n1, H1, N1) / N1 + moda(n2, H2, N2) / N2 +
                       moda(n3, H3, N3) / N3);
    v = {std::cos(phase) * norm, std::sin(phase) * norm};
  });
}

template <typename T1, typename T2>
sycl::event transpose_tile(std::size_t m, std::size_t n, T1 in, std::size_t lda,
                           T2 out, std::size_t ldb,
                           const std::vector<sycl::event> &events = {}) {
  auto &&q = dr::sp::__detail::get_queue_for_pointer(out);
  constexpr std::size_t tile_size = 16;
  const std::size_t m_max = ((m + tile_size - 1) / tile_size) * tile_size;
  const std::size_t n_max = ((n + tile_size - 1) / tile_size) * tile_size;
  using temp_t = std::iter_value_t<T1>;
  const auto in_ = in.get_raw_pointer();

  return q.submit([&](sycl::handler &cgh) {
    cgh.depends_on(events);
    sycl::local_accessor<temp_t, 2> tile(
        sycl::range<2>(tile_size, tile_size + 1), cgh);

    cgh.parallel_for(sycl::nd_range<2>{{m_max, n_max}, {tile_size, tile_size}},
                     [=](sycl::nd_item<2> item) {
                       unsigned x = item.get_global_id(1);
                       unsigned y = item.get_global_id(0);
                       unsigned xth = item.get_local_id(1);
                       unsigned yth = item.get_local_id(0);

                       if (x < n && y < m)
                         tile[yth][xth] = in_[(y)*lda + x];
                       item.barrier(sycl::access::fence_space::local_space);

                       x = item.get_group(0) * tile_size + xth;
                       y = item.get_group(1) * tile_size + yth;
                       if (x < m && y < n)
                         out[(y)*ldb + x] = tile[xth][yth];
                     });
  });
}

template <typename T>
void transpose_matrix(dr::sp::distributed_dense_matrix<T> &i_mat,
                      dr::sp::distributed_dense_matrix<T> &o_mat) {
  std::vector<sycl::event> events;
  int ntiles = i_mat.segments().size();
  int lda = i_mat.shape()[1];
  int ldb = o_mat.shape()[1];
  // need to handle offsets better
  int m_local = i_mat.shape()[0] / ntiles;
  int n_local = o_mat.shape()[0] / ntiles;
  for (int i = 0; i < ntiles; i++) {
    for (int j_ = 0; j_ < ntiles; j_++) {
      int j = (j_ + i) % ntiles;
      auto &&send_tile = i_mat.tile({i, 0});
      auto &&recv_tile = o_mat.tile({j, 0});
      auto e = transpose_tile(m_local, n_local, send_tile.data() + j * n_local,
                              lda, recv_tile.data() + i * m_local, ldb);
      events.push_back(e);
    }
  }
  sycl::event::wait(events);
}

sycl::event mkl_dft(auto aplan_, auto data_, bool forward) {
  if (forward) {
    return oneapi::mkl::dft::compute_forward(*aplan_, data_);
  }
  return oneapi::mkl::dft::compute_backward(*aplan_, data_);
}

template <typename T> struct dft_precision {
  static const oneapi::mkl::dft::precision value =
      oneapi::mkl::dft::precision::SINGLE;
};

template <> struct dft_precision<double> {
  static const oneapi::mkl::dft::precision value =
      oneapi::mkl::dft::precision::DOUBLE;
};

template <typename T> class distributed_fft {
  std::int64_t m_, n_;
  dr::sp::block_cyclic row_blocks_;
  dr::sp::distributed_dense_matrix<value_t> in_, out_;

  using fft_plan_t =
      oneapi::mkl::dft::descriptor<dft_precision<T>::value,
                                   oneapi::mkl::dft::domain::COMPLEX>;

  std::vector<fft_plan_t *> fft_plans;

  void fft_threads(dr::sp::distributed_dense_matrix<std::complex<T>> &i_mat,
                   dr::sp::distributed_dense_matrix<std::complex<T>> &o_mat,
                   bool forward) {
    int ntiles = i_mat.segments().size();
    std::size_t mask_p = 0, mask_u = 1;
    if (!forward)
      std::swap(mask_p, mask_u);

    std::vector<std::jthread> threads;
    std::latch fft_phase_done(ntiles);

    for (int i = 0; i < ntiles; ++i) {
      threads.emplace_back([i, ntiles, forward,
                            plan_0 = fft_plans[2 * i + mask_p],
                            plan_1 = fft_plans[2 * i + mask_u], &fft_phase_done,
                            &i_mat, &o_mat] {
        auto &&atile = i_mat.tile({i, 0});
        mkl_dft(plan_0, dr::sp::__detail::local(atile).data(), forward).wait();

        fft_phase_done.arrive_and_wait();

        int lda = i_mat.shape()[1];
        int ldb = o_mat.shape()[1];
        int m_local = i_mat.shape()[0] / ntiles;
        int n_local = o_mat.shape()[0] / ntiles;

        std::vector<sycl::event> events;
        for (int j_ = 0; j_ < ntiles; j_++) {
          int j = (j_ + i) % ntiles;
          auto &&recv_tile = o_mat.tile({j, 0});
          auto e = transpose_tile(m_local, n_local, atile.data() + j * n_local,
                                  lda, recv_tile.data() + i * m_local, ldb);
          events.push_back(e);
        }
        sycl::event::wait(events);

        auto &&btile = o_mat.tile({i, 0});
        mkl_dft(plan_1, dr::sp::__detail::local(btile).data(), forward).wait();
      });
    }
  }

  void fft_serial(dr::sp::distributed_dense_matrix<std::complex<T>> &i_mat,
                  dr::sp::distributed_dense_matrix<std::complex<T>> &o_mat,
                  bool forward) {

    std::size_t mask_p = 0, mask_u = 1;
    if (!forward)
      std::swap(mask_p, mask_u);

    std::vector<sycl::event> events;
    int nprocs = i_mat.segments().size();
    for (int i = 0; i < nprocs; i++) {
      auto &&atile = i_mat.tile({i, 0});
      auto e = oneapi::mkl::dft::compute_forward(
          *fft_plans[2 * i + mask_p], dr::sp::__detail::local(atile).data());
      events.push_back(e);
    }
    sycl::event::wait(events);
    events.clear();

    transpose_matrix(i_mat, o_mat);

    for (int i = 0; i < nprocs; i++) {
      auto &&atile = o_mat.tile({i, 0});
      auto e = oneapi::mkl::dft::compute_forward(
          *fft_plans[2 * i + mask_u], dr::sp::__detail::local(atile).data());
      events.push_back(e);
    }
    sycl::event::wait(events);
  }

public:
  explicit distributed_fft(std::int64_t m_in)
      : m_(dr::__detail::round_up(m_in, dr::sp::nprocs())), n_(m_ * m_),
        row_blocks_({dr::sp::tile::div, dr::sp::tile::div},
                    {dr::sp::nprocs(), 1}),
        in_({m_, n_}, row_blocks_), out_({n_, m_}, row_blocks_) {
    auto nprocs = dr::sp::nprocs();
    fft_plans.reserve(2 * nprocs);

    int m_local = m_ / nprocs;
    for (int i = 0; i < nprocs; i++) {
      auto &&q = dr::sp::__detail::queue(i);
      fft_plan_t *desc = new fft_plan_t({m_, m_});
      desc->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS,
                      m_local);
      desc->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, m_ * m_);
      desc->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, m_ * m_);
      desc->set_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE,
                      (1.0 / (m_ * m_ * m_)));
      // show_plan("yz", desc);
      desc->commit(q);
      fft_plans.emplace_back(desc);

      desc = new fft_plan_t(m_);
      desc->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS,
                      (m_ * m_) / nprocs);
      desc->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, m_);
      desc->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, m_);
      // show_plan("x", desc);
      desc->commit(q);
      fft_plans.emplace_back(desc);
    }

    fmt::print("Initializing...\n");
    init_matrix_3d(in_, m_);
  }

  ~distributed_fft() {
    int i = fft_plans.size() - 1;
    while (i >= 0) {
      delete fft_plans[i];
      --i;
    }
  }

  void compute() {
    fft_threads(in_, out_, true);
    fft_threads(out_, in_, false);
  }

  void check() {
    fmt::print("Testing\n");
    dr::sp::distributed_dense_matrix<value_t> t_mat({m_, n_}, row_blocks_);

    fft_threads(in_, out_, true);
    fft_threads(out_, t_mat, false);

    fmt::print("Checking results\n");
    auto sub_view = dr::sp::views::zip(values_view(in_), values_view(t_mat)) |
                    dr::sp::views::transform([](auto &&e) {
                      auto &&[value, ref] = e;
                      return value - ref;
                    });
    auto diff_sum = dr::sp::reduce(dr::sp::par_unseq, sub_view, value_t{});
    fmt::print("Difference {} {} \n", diff_sum.real(), diff_sum.imag());
  }

  void show_plan(const std::string &title, auto *plan) {
    MKL_LONG transforms, fwd_distance, bwd_distance;
    real_t forward_scale, backward_scale;
    plan->get_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS,
                    &transforms);
    plan->get_value(oneapi::mkl::dft::config_param::FWD_DISTANCE,
                    &fwd_distance);
    plan->get_value(oneapi::mkl::dft::config_param::BWD_DISTANCE,
                    &bwd_distance);
    plan->get_value(oneapi::mkl::dft::config_param::FORWARD_SCALE,
                    &forward_scale);
    plan->get_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE,
                    &backward_scale);
    fmt::print("{}:\n  number of transforms: {}\n  fwd distance: {}\n  bwd "
               "distance: {}\n  forward scale: {}\n  backward scale: {}\n",
               title, transforms, fwd_distance, bwd_distance, forward_scale,
               backward_scale);
  }
};

#ifdef STANDALONE_BENCHMARK

int main(int argc, char *argv[]) {

  cxxopts::Options options_spec(argv[0], "fft3d");
  // clang-format off
  options_spec.add_options()
    ("d, num-devices", "number of sycl devices, 0 uses all available devices", cxxopts::value<std::size_t>()->default_value("0"))
    ("n", "problem size", cxxopts::value<std::size_t>()->default_value("8"))
    ("r,repetitions", "Number of repetitions", cxxopts::value<std::size_t>()->default_value("0"))
    ("log", "enable logging")
    ("logprefix", "appended .RANK.log", cxxopts::value<std::string>()->default_value("dr"))
    ("verbose", "verbose output")
    ("h,help", "Print help");
  // clang-format on

  cxxopts::ParseResult options;
  try {
    options = options_spec.parse(argc, argv);
  } catch (const cxxopts::OptionParseException &e) {
    std::cout << options_spec.help() << "\n";
    exit(1);
  }

  std::unique_ptr<std::ofstream> logfile;
  if (options.count("log")) {
    logfile.reset(new std::ofstream(
        fmt::format("{}.log", options["logprefix"].as<std::string>())));
    dr::drlog.set_file(*logfile);
  }

  // 512^3 up to 3072
  std::size_t dim = options["n"].as<std::size_t>();
  std::size_t nreps = options["repetitions"].as<std::size_t>();

  auto devices = dr::sp::get_numa_devices(sycl::default_selector_v);
  std::size_t ranks = options["num-devices"].as<std::size_t>();
  if (ranks != 0 && ranks < devices.size()) {
    devices.resize(ranks);
  }
  for (auto device : devices) {
    fmt::print("Device: {}\n", device.get_info<sycl::info::device::name>());
  }
  dr::sp::init(devices);
  distributed_fft<real_t> fft3d(dim);

  if (nreps == 0) {
    fft3d.check();
  } else {
    for (int iter = 0; iter < nreps; ++iter) {
      fft3d.compute();
    }
  }

  return 0;
}

#else

static void FFT3D_DR(benchmark::State &state) {
  std::size_t dim = check_results ? 8 : 768;

  distributed_fft<real_t> fft3d(dim);

  for (auto _ : state) {
    fft3d.compute();
  }
}

DR_BENCHMARK(FFT3D_DR);

#endif // STANDALONE_BENCHMARK
