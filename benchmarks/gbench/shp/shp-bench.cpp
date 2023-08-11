// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "../common/dr_bench.hpp"

std::size_t default_vector_size;
std::size_t default_repetitions;
bool weak_scaling;

std::size_t comm_rank = 0;
std::size_t comm_size = 1;
std::size_t ranks = 0;

cxxopts::ParseResult options;

int main(int argc, char *argv[]) {
  benchmark::Initialize(&argc, argv);

  cxxopts::Options options_spec(argv[0], "DR SHP tests");

  // clang-format off
  options_spec.add_options()
    ("d, num-devices", "number of sycl devices, 0 uses all available devices", cxxopts::value<std::size_t>()->default_value("0"))
    ("drhelp", "Print help")
    ("reps", "Debug repetitions for short duration vector operations", cxxopts::value<std::size_t>()->default_value("1"))
    ("vector-size", "Default vector size", cxxopts::value<std::size_t>()->default_value("100000000"))
    ("context", "Additional google benchmark context", cxxopts::value<std::vector<std::string>>())
    ("weak-scaling", "Scale the vector size by the number of ranks", cxxopts::value<bool>()->default_value("false"))
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

  default_vector_size = options["vector-size"].as<std::size_t>();
  default_repetitions = options["reps"].as<std::size_t>();
  ranks = options["num-devices"].as<std::size_t>();
  weak_scaling = options["weak-scaling"].as<bool>();

  auto available_devices = dr::shp::get_numa_devices(sycl::default_selector_v);
  if (ranks == 0) {
    ranks = available_devices.size();
  }

  if (options["weak-scaling"].as<bool>())
    default_vector_size = default_vector_size * ranks;

  add_configuration(0, options);

  std::vector<sycl::device> devices;
  for (std::size_t i = 0; i < ranks; i++) {
    devices.push_back(available_devices[i % available_devices.size()]);
    benchmark::AddCustomContext("device_info" + std::to_string(i),
                                device_info(devices.back()));
  }

  dr::shp::init(devices);

  benchmark::RunSpecifiedBenchmarks();

  benchmark::Shutdown();

  return 0;
}
