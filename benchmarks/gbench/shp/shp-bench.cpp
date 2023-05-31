// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-bench.hpp"

std::size_t default_vector_size;
std::size_t default_repetitions;

std::size_t comm_rank = 0;
std::size_t comm_size = 1;

cxxopts::ParseResult options;

int main(int argc, char *argv[]) {
  benchmark::Initialize(&argc, argv);

  cxxopts::Options options_spec(argv[0], "DR SHP tests");

  // clang-format off
  options_spec.add_options()
    ("d, num-devices", "number of sycl devices, 0 uses all available devices", cxxopts::value<unsigned int>()->default_value("0"))
    ("drhelp", "Print help")
    ("reps", "Debug repetitions for short duration vector operations", cxxopts::value<std::size_t>()->default_value("1"))
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

  default_vector_size = options["vector-size"].as<std::size_t>();
  default_repetitions = options["reps"].as<std::size_t>();
  const unsigned int dev_num = options["num-devices"].as<unsigned int>();
  fmt::print("Configuration:\n"
             "  default vector size: {}\n"
             "  default repetitions: {}\n"
             "  number of devices requested: {}\n",
             default_vector_size, default_repetitions, dev_num);

  auto devices = dr::shp::get_numa_devices(sycl::default_selector_v);

  if (dev_num > 0) {
    unsigned int i = 0;
    while (devices.size() < dev_num) {
      devices.push_back(devices[i++]);
    }
    devices.resize(dev_num); // if too many devices
  }
  for (auto &device : devices) {
    fmt::print("    {}\n", device.get_info<sycl::info::device::name>());
  }

  dr::shp::init(devices);

  benchmark::RunSpecifiedBenchmarks();

  benchmark::Shutdown();

  return 0;
}
