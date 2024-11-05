// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "cxxopts.hpp"
#include "fmt/core.h"
#include "mpi.h"
#include "oneapi/mkl/dft.hpp"
#include <complex>

#include "dr/mp.hpp"

#ifdef STANDALONE_BENCHMARK

MPI_Comm comm;
int comm_rank;
int comm_size;

#else

#include "../common/dr_bench.hpp"

#endif

using real_t = float;
using value_t = std::complex<real_t>;

bool verbose = false;

template <typename Base>
struct fmt::formatter<std::complex<Base>, char>
    : public formatter<string_view> {
  template <typename FmtContext>
  auto format(std::complex<Base> c, FmtContext &ctx) const {
    fmt::format_to(ctx.out(), "{}+{}i", c.real(), c.imag());
    return ctx.out();
  }
};

// Adapted examples/sycl/dft/source/dp_complex_3d.cpp
template <typename T>
void init_matrix(dr::mp::distributed_mdarray<std::complex<T>, 3> &mat) {

  constexpr T TWOPI = 6.2831853071795864769;
  constexpr int H1 = -1, H2 = -2, H3 = -3;
  int N1 = mat.extent(2), N2 = mat.extent(1), N3 = mat.extent(0);

  auto moda = [](int K, int L, int M) { return (T)(((long long)K * L) % M); };

  const T norm = T(1) / (N3 * N2 * N1);

  auto init = [=](auto index, auto v) {
    auto phase = TWOPI * (moda(static_cast<int>(index[2]), H1, N1) / N1 +
                          moda(static_cast<int>(index[1]), H2, N2) / N2 +
                          moda(static_cast<int>(index[0]), H3, N3) / N3);
    std::get<0>(v) = {std::cos(phase) * norm, std::sin(phase) * norm};
  };

  dr::mp::for_each(init, mat);
}

template <typename T> struct dft_precision {
  static const oneapi::mkl::dft::precision value =
      oneapi::mkl::dft::precision::SINGLE;
};

template <> struct dft_precision<double> {
  static const oneapi::mkl::dft::precision value =
      oneapi::mkl::dft::precision::DOUBLE;
};

void show_plan(const std::string &title, auto *plan) {
  MKL_LONG transforms, fwd_distance, bwd_distance;
  real_t forward_scale, backward_scale;
  plan->get_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS,
                  &transforms);
  plan->get_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, &fwd_distance);
  plan->get_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, &bwd_distance);
  plan->get_value(oneapi::mkl::dft::config_param::FORWARD_SCALE,
                  &forward_scale);
  plan->get_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE,
                  &backward_scale);
  fmt::print("{}:\n  number of transforms: {}\n  fwd distance: {}\n  bwd "
             "distance: {}\n  forward scale: {}\n  backward scale: {}\n",
             title, transforms, fwd_distance, bwd_distance, forward_scale,
             backward_scale);
}

template <typename T> class distributed_fft {
public:
  explicit distributed_fft(std::size_t x, std::size_t y, std::size_t z)
      : i_mat_({x, y, z}), o_mat_({y, z, x}) {
    sycl::queue q = dr::mp::select_queue();
    i_slab_ = *(dr::mp::local_mdspans(i_mat_).begin());
    o_slab_ = *(dr::mp::local_mdspans(o_mat_).begin());

    std::int64_t i_x = i_slab_.extent(0), i_y = i_slab_.extent(1),
                 i_z = i_slab_.extent(2);
    fft_2d_plan_ = new fft_plan_t({i_y, i_z});
    fft_2d_plan_->set_value(
        oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, i_x);
    fft_2d_plan_->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE,
                            i_y * i_z);
    fft_2d_plan_->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE,
                            i_y * i_z);
    fft_2d_plan_->set_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE,
                            (1.0 / (x * i_y * i_z)));
    fft_2d_plan_->commit(q);
    // show_plan("yz", fft_2d_plan_);

    auto o_x = o_slab_.extent(0);
    auto o_y = o_slab_.extent(1);
    auto o_z = o_slab_.extent(2);
    fft_1d_plan_ = new fft_plan_t(o_z);
    fft_1d_plan_->set_value(
        oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, o_x * o_y);
    fft_1d_plan_->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, o_z);
    fft_1d_plan_->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, o_z);
    fft_1d_plan_->commit(q);
    // show_plan("x", fft_1d_plan_);

    init_matrix(i_mat_);
  }

  void check() {
    auto extents = i_mat_.mdspan().extents();
    mat t_mat({extents.extent(0), extents.extent(1), extents.extent(2)});
    init_matrix(t_mat);
    compute();

    auto sub_view = dr::mp::views::zip(i_mat_, t_mat) |
                    dr::mp::views::transform([](auto &&e) {
                      auto [a, b] = e;
                      return value_t(a) - value_t(b);
                    });

    auto diff_sum = dr::mp::reduce(sub_view, value_t{});
    if (comm_rank == 0) {
      if (verbose) {
        print_matrix("i_mat", i_mat_);
        print_matrix("t_mat", t_mat);
      }
      fmt::print("Difference {}\n", diff_sum);
    }
  }

  void compute() {
    oneapi::mkl::dft::compute_forward(*fft_2d_plan_, i_slab_.data_handle())
        .wait();
    dr::mp::transpose(i_mat_, o_mat_, true);

    oneapi::mkl::dft::compute_forward(*fft_1d_plan_, o_slab_.data_handle())
        .wait();

    oneapi::mkl::dft::compute_backward(*fft_1d_plan_, o_slab_.data_handle())
        .wait();
    dr::mp::transpose(o_mat_, i_mat_, false);
    oneapi::mkl::dft::compute_backward(*fft_2d_plan_, i_slab_.data_handle())
        .wait();
  }

private:
  static void print_matrix(auto title, const auto &mat) {
    fmt::print("{}:\n", title);

    auto m = mat.mdspan();
    for (std::size_t i = 0; i < m.extent(0); i++) {
      for (std::size_t j = 0; j < m.extent(1); j++) {
        for (std::size_t k = 0; k < m.extent(2); k++) {
          fmt::print("{} ", value_t(m(i, j, k)));
        }
        fmt::print("\n");
      }
      fmt::print("\n");
    }
  }

  using fft_plan_t =
      oneapi::mkl::dft::descriptor<dft_precision<T>::value,
                                   oneapi::mkl::dft::domain::COMPLEX>;
  fft_plan_t *fft_1d_plan_;
  fft_plan_t *fft_2d_plan_;

  using mat = dr::mp::distributed_mdarray<value_t, 3>;
  mat i_mat_, o_mat_;

  using slab = decltype(*(dr::mp::local_mdspans(i_mat_).begin()));
  slab i_slab_, o_slab_;
};

#ifdef STANDALONE_BENCHMARK

void fft(std::size_t nreps, std::size_t x, std::size_t y, std::size_t z) {
  distributed_fft<real_t> fft3d(x, y, z);

  if (nreps == 0) {
    fft3d.check();
  } else {
    for (int iter = 0; iter < nreps; ++iter) {
      fft3d.compute();
    }
  }
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &comm_rank);
  MPI_Comm_size(comm, &comm_size);

  cxxopts::Options options_spec(argv[0], "fft3d");
  // clang-format off
  options_spec.add_options()
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
    logfile.reset(new std::ofstream(options["logprefix"].as<std::string>() +
                                    fmt::format(".{}.log", comm_rank)));
    dr::drlog.set_file(*logfile);
  }

  if (options.count("verbose")) {
    verbose = true;
  }

  // 512^3 up to 3072
  std::size_t x = options["n"].as<std::size_t>();
  std::size_t y = x;
  std::size_t z = x;
  std::size_t nreps = options["repetitions"].as<std::size_t>();

  dr::mp::init(dr::mp::select_queue());
  fft(nreps, x, y, z);

  dr::mp::finalize();
  MPI_Finalize();
  return 0;
}

#else

static void FFT3D_DR(benchmark::State &state) {
  // fft requires usm shared allocation
  assert(dr::mp::use_sycl());

  std::size_t x = 768;
  std::size_t y = x;
  std::size_t z = x;
  if (check_results) {
    x = y = z = 8;
  }

  distributed_fft<real_t> fft3d(x, y, z);

  for (auto _ : state) {
    fft3d.compute();
  }
}

DR_BENCHMARK(FFT3D_DR);

#endif
