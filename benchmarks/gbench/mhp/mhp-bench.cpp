// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-bench.hpp"

MPI_Comm comm;
std::size_t comm_rank;
std::size_t comm_size;

std::size_t default_vector_size;
std::size_t default_repetitions;
std::size_t stencil_steps;

cxxopts::ParseResult options;

// This reporter does nothing.
// We can use it to disable output from all but the root process
class NullReporter : public ::benchmark::BenchmarkReporter {
public:
  NullReporter() {}
  virtual bool ReportContext(const Context &) { return true; }
  virtual void ReportRuns(const std::vector<Run> &) {}
  virtual void Finalize() {}
};

void dr_init() {
#ifdef SYCL_LANGUAGE_VERSION
  if (options.count("sycl")) {
    sycl::queue q;
    if (comm_rank == 0) {
      fmt::print("  run on sycl device: {}\n",
                 q.get_device().get_info<sycl::info::device::name>());
    }
    dr::mhp::init(q);
    return;
  }
#endif

  if (comm_rank == 0) {
    fmt::print("  run on CPU\n");
  }
  dr::mhp::init();
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  comm = MPI_COMM_WORLD;
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  comm_rank = rank;
  comm_size = size;

  benchmark::Initialize(&argc, argv);

  cxxopts::Options options_spec(argv[0], "DR MHP tests");

  // clang-format off
  options_spec.add_options()
    ("drhelp", "Print help")
    ("log", "Enable logging")
#ifdef SYCL_LANGUAGE_VERSION
    ("sycl", "Execute on SYCL device")
#endif
    ("reps", "Debug repetitions for short duration vector operations", cxxopts::value<std::size_t>()->default_value("1"))
    ("stencil-steps", "Default steps for stencil", cxxopts::value<std::size_t>()->default_value("100"))
    ("vector-size", "Default vector size", cxxopts::value<std::size_t>()->default_value("100000000"))
    ;
  // clang-format on

  try {
    options = options_spec.parse(argc, argv);
  } catch (const cxxopts::OptionParseException &e) {
    std::cout << options_spec.help() << "\n";
    exit(1);
  }

  if (options.count("drhelp")) {
    std::cout << options_spec.help() << "\n";
    exit(0);
  }

  std::ofstream *logfile = nullptr;
  if (options.count("log")) {
    logfile = new std::ofstream(fmt::format("dr.{}.log", comm_rank));
    dr::drlog.set_file(*logfile);
  }
  dr::drlog.debug("Rank: {}\n", comm_rank);

  default_vector_size = options["vector-size"].as<std::size_t>();
  default_repetitions = options["reps"].as<std::size_t>();
  stencil_steps = options["stencil-steps"].as<std::size_t>();
  if (comm_rank == 0) {
    fmt::print("Configuration:\n"
               "  default vector size: {}\n"
               "  default repetitions: {}\n",
               default_vector_size, default_repetitions);
  }

  dr_init();
  if (rank == 0) {
    benchmark::RunSpecifiedBenchmarks();
  } else {
    NullReporter null_reporter;
    benchmark::RunSpecifiedBenchmarks(&null_reporter);
  }
  benchmark::Shutdown();

  if (logfile) {
    delete logfile;
  }

  MPI_Finalize();

  return 0;
}
