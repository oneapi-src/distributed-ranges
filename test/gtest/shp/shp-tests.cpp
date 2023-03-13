// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "shp-tests.hpp"

using TestTypes = ::testing::Types<shp::distributed_vector<int>,
                                   shp::distributed_vector<float>>;

#include "common/distributed_vector.hpp"
#include "common/drop.hpp"
// Not implemented???
// #include "common/fill.hpp"
#include "common/for_each.hpp"
#include "common/reduce.hpp"
#include "common/subrange.hpp"
#include "common/take.hpp"
#include "common/transform_view.hpp"
// Issue with 2 element zips
// #include "common/zip.hpp"

// To share tests with MHP
std::size_t comm_rank = 0;
std::size_t comm_size = 1;

cxxopts::ParseResult options;

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  cxxopts::Options options_spec(argv[0], "DR SHP tests");

  // clang-format off
  options_spec.add_options()
    ("drhelp", "Print help")
    ("d, devicesCount", "number of GPUs to create", cxxopts::value<unsigned int>()->default_value("0"));
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

  const unsigned int dev_num = options["devicesCount"].as<unsigned int>();
  auto devices = shp::get_numa_devices(sycl::default_selector_v);

  if (dev_num > 0) {
    unsigned int i = 0;
    while (devices.size() < dev_num)
      devices.push_back(devices[i++]);
    devices.resize(dev_num); // if too many devices
  }

  shp::init(devices);

  for (auto &device : devices)
    std::cout << "  Device: " << device.get_info<sycl::info::device::name>()
              << "\n";

  return RUN_ALL_TESTS();
}
