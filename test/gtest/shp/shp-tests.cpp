// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "shp-tests.hpp"

cxxopts::ParseResult options;

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);

  cxxopts::Options options_spec(argv[0], "DR SHP tests");

  // clang-format off
  options_spec.add_options()
    ("drhelp", "Print help");
  // clang-format on

  try {
    options = options_spec.parse(argc, argv);
  } catch (const cxxopts::OptionParseException &e) {
    fmt::print("{}\n", options_spec.help());
    exit(1);
  }

  if (options.count("drhelp")) {
    fmt::print("{}\n", options_spec.help());
    exit(0);
  }

  auto devices = shp::get_numa_devices(sycl::default_selector_v);

  std::cout << " *** Running shp-tests for 1 device(s) ***\n";
  shp::init(devices);

  for (auto &device : devices) {
    std::cout << "  Device: " << device.get_info<sycl::info::device::name>()
              << "\n";
  }

  auto res = RUN_ALL_TESTS();
  shp::finalize();

  auto d0 = devices[0];
  for (size_t nd = 2; nd <= 7; nd++) {
    std::cout << " *** Running shp-tests for " << nd << " device(s) ***\n";
    devices.push_back(d0);
    shp::init(devices);

    for (auto &device : devices) {
      std::cout << "  Device: " << device.get_info<sycl::info::device::name>()
                << "\n";
    }

    auto res = RUN_ALL_TESTS();
    shp::finalize();
  }

  return 0;
}
