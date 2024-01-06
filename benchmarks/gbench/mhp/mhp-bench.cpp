// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "../common/dr_bench.hpp"

MPI_Comm comm;
std::size_t comm_rank;
std::size_t ranks;

std::size_t default_vector_size;
std::size_t default_repetitions;
std::size_t stencil_steps;
std::size_t num_rows;
std::size_t num_columns;
bool check_results;
bool weak_scaling;
bool device_memory;

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
    sycl::queue q = dr::mhp::select_queue(options.count("different-devices"));
    benchmark::AddCustomContext("device_info", device_info(q.get_device()));
    dr::mhp::init(q, device_memory ? sycl::usm::alloc::device
                                   : sycl::usm::alloc::shared);
    return;
  }
#endif

  if (comm_rank == 0) {
    fmt::print("  run on: CPU\n");
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
  ranks = size;

  bool dry_run = false;
  for (int i = 0; i < argc; ++i) {
    auto param = std::string(argv[i]);
    if (comm_rank != 0 && param.starts_with("--benchmark_out=")) {
      // Only rank 0 does file output
      *argv[i] = 0;
    }
    // do not initialize if only lists tests
    if (param.starts_with("--benchmark_list_tests") &&
        !param.starts_with("--benchmark_list_tests=false")) {
      dry_run = true;
    }
  }

  benchmark::Initialize(&argc, argv);

  cxxopts::Options options_spec(argv[0], "DR MHP tests");

  // clang-format off
  options_spec.add_options()
    ("check", "Check results")
    ("columns", "Number of columns", cxxopts::value<std::size_t>()->default_value("10000"))
    ("drhelp", "Print help")
    ("log", "Enable logging")
    ("logprefix", "appended .RANK.log", cxxopts::value<std::string>()->default_value("dr"))
    ("log-filter", "Filter the log", cxxopts::value<std::vector<std::string>>())
#ifdef SYCL_LANGUAGE_VERSION
    ("sycl", "Execute on SYCL device")
    ("different-devices", "ensure no multiple ranks on one device")
#endif
    ("reps", "Debug repetitions for short duration vector operations", cxxopts::value<std::size_t>()->default_value("1"))
    ("rows", "Number of rows", cxxopts::value<std::size_t>()->default_value("10000"))
    ("stencil-steps", "Default steps for stencil", cxxopts::value<std::size_t>()->default_value("10"))
    ("vector-size", "Default vector size", cxxopts::value<std::size_t>()->default_value("100000000"))
    ("context", "Additional google benchmark context", cxxopts::value<std::vector<std::string>>())
    ("device-memory", "Use device memory")
    ("weak-scaling", "Scale the vector size by the number of ranks", cxxopts::value<bool>()->default_value("false"))
    ("benchmark_list_tests", "only lists tests, skip backends initialization")
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

  std::unique_ptr<std::ofstream> logfile;
  if (options.count("log")) {
    logfile.reset(new std::ofstream(options["logprefix"].as<std::string>() +
                                    fmt::format(".{}.log", comm_rank)));
    dr::drlog.set_file(*logfile);
    if (options.count("log-filter")) {
      dr::drlog.filter(options["log-filter"].as<std::vector<std::string>>());
    }
  }
  dr::drlog.debug("Rank: {}\n", comm_rank);

  default_vector_size = options["vector-size"].as<std::size_t>();
  default_repetitions = options["reps"].as<std::size_t>();
  stencil_steps = options["stencil-steps"].as<std::size_t>();
  num_rows = options["rows"].as<std::size_t>();
  num_columns = options["columns"].as<std::size_t>();
  check_results = options.count("check");
  weak_scaling = options["weak-scaling"].as<bool>();
  device_memory = options["device-memory"].as<bool>();

  if (weak_scaling)
    default_vector_size = default_vector_size * ranks;

  add_configuration(comm_rank, options);

  if (!dry_run)
    dr_init();

  if (rank == 0) {
    benchmark::RunSpecifiedBenchmarks();
  } else {
    // Disable console output if not rank 0
    NullReporter null_reporter;
    benchmark::RunSpecifiedBenchmarks(&null_reporter);
  }
  benchmark::Shutdown();

  if (!dry_run)
    dr::mhp::finalize();
  MPI_Finalize();

  return 0;
}
