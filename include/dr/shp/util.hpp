#pragma once

#include <CL/sycl.hpp>
#include <iostream>
#include <ranges>

namespace shp {

template <typename Selector>
cl::sycl::device select_device(Selector &&selector) {
  cl::sycl::device d;

  try {
    d = cl::sycl::device(std::forward<Selector>(selector));
    std::cout << "Running on device \""
              << d.get_info<cl::sycl::info::device::name>() << "\""
              << std::endl;
  } catch (cl::sycl::exception const &e) {
    std::cout << "Cannot select an accelerator\n" << e.what() << "\n";
    std::cout << "Using a CPU device\n";
    d = cl::sycl::device(cl::sycl::cpu_selector());
  }
  return d;
}

template <typename Selector> void list_devices(Selector &&selector) {
  namespace sycl = cl::sycl;

  sycl::platform p(std::forward<Selector>(selector));
  auto devices = p.get_devices();

  printf("--Platform Info-----------------\n");

  printf("Platform %s has %lu root devices.\n",
         p.get_info<sycl::info::platform::name>().c_str(), devices.size());

  for (size_t i = 0; i < devices.size(); i++) {
    auto &&device = devices[i];

    printf("  %lu %s\n", i,
           device.get_info<sycl::info::device::name>().c_str());

    using namespace sycl::info;
    auto subdevices = device.create_sub_devices<
        partition_property::partition_by_affinity_domain>(
        partition_affinity_domain::numa);

    printf("   Subdevices:\n");
    for (size_t j = 0; j < subdevices.size(); j++) {
      auto &&subdevice = subdevices[j];
      printf("     %lu.%lu %s\n", i, j,
             subdevice.get_info<sycl::info::device::name>().c_str());
    }
  }

  printf("--------------------------------\n");
}

template <typename Selector>
std::vector<cl::sycl::device> get_numa_devices(Selector &&selector) {
  namespace sycl = cl::sycl;

  std::vector<sycl::device> devices;

  sycl::platform p(std::forward<Selector>(selector));
  auto root_devices = p.get_devices();

  for (auto &&root_device : root_devices) {
    using namespace sycl::info;
    auto subdevices = root_device.create_sub_devices<
        partition_property::partition_by_affinity_domain>(
        partition_affinity_domain::numa);

    for (auto &&subdevice : subdevices) {
      devices.push_back(subdevice);
    }
  }

  return devices;
}

template <typename Selector>
std::vector<cl::sycl::device> get_devices(Selector &&selector) {
  namespace sycl = cl::sycl;

  sycl::platform p(std::forward<Selector>(selector));
  return p.get_devices();
}

template <typename Range> void print_range(Range &&r, std::string label = "") {
  size_t indent = 1;

  if (label != "") {
    std::cout << "\"" << label << "\": ";
    indent += label.size() + 4;
  }

  std::string indent_whitespace(indent, ' ');

  std::cout << "[";
  size_t columns = 10;
  size_t count = 1;
  for (auto iter = r.begin(); iter != r.end(); ++iter) {
    std::cout << static_cast<std::ranges::range_value_t<Range>>(*iter);

    auto next = iter;
    ++next;
    if (next != r.end()) {
      std::cout << ", ";
      if (count % columns == 0) {
        std::cout << "\n" << indent_whitespace;
      }
    }
    ++count;
  }
  std::cout << "]" << std::endl;
}

} // namespace shp
