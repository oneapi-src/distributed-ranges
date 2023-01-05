// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

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

inline void list_devices() {
  auto platforms = sycl::platform::get_platforms();

  for (auto &platform : platforms) {
    std::cout << "Platform: " << platform.get_info<sycl::info::platform::name>()
              << std::endl;

    auto devices = platform.get_devices();
    for (auto &device : devices) {
      std::cout << "  Device: " << device.get_info<sycl::info::device::name>()
                << std::endl;
    }
  }
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
std::vector<cl::sycl::device> get_numa_devices_impl_(Selector &&selector) {
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

template <typename Selector>
std::vector<cl::sycl::device> get_numa_devices(Selector &&selector) {
  try {
    return get_numa_devices_impl_(std::forward<Selector>(selector));
  } catch (cl::sycl::feature_not_supported) {
    std::cerr << "NUMA partitioning not supported, returning root devices..."
              << std::endl;
    return get_devices(std::forward<Selector>(selector));
  }
}

// Return exactly `n` devices obtained using the selector `selector`.
// May duplicate devices
template <typename Selector>
std::vector<cl::sycl::device> get_duplicated_devices(Selector &&selector,
                                                     std::size_t n) {
  auto devices = get_numa_devices(std::forward<Selector>(selector));

  if (devices.size() >= n) {
    return std::vector<cl::sycl::device>(devices.begin(), devices.begin() + n);
  } else {
    std::size_t i = 0;
    while (devices.size() < n) {
      auto d = devices[i++];
      devices.push_back(d);
    }
    return devices;
  }
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

template <typename R> void print_range_details(R &&r, std::string label = "") {
  if (label != "") {
    std::cout << "\"" << label << "\" ";
  }

  std::cout << "distributed range with " << r.segments().size() << " segments."
            << std::endl;

  std::size_t idx = 0;
  for (auto &&segment : r.segments()) {
    std::cout << "Seg " << idx++ << ", size " << segment.size() << " (rank "
              << segment.rank() << ")" << std::endl;
  }
}

// Allocate spans on a number of devices.

template <typename T>
auto allocate_device_span(std::size_t size, std::size_t rank,
                          cl::sycl::context context, auto &&devices) {
  auto data = shp::device_allocator<T>(context, devices[rank]).allocate(size);

  return shp::device_span<T, decltype(data)>(data, size, rank);
}

// Return a range of spans, with one span allocated on each device using
// device memory.
template <typename T>
auto allocate_device_spans(std::size_t size, cl::sycl::context context,
                           auto &&devices) {
  std::vector<shp::device_span<T, shp::device_ptr<T>>> spans;
  for (size_t rank = 0; rank < devices.size(); rank++) {
    spans.push_back(allocate_device_span<T>(size, rank, context, devices));
  }
  return spans;
}

template <typename T>
shp::device_span<T> allocate_shared_span(std::size_t size, std::size_t rank,
                                         auto &&devices) {
  cl::sycl::queue q(devices[rank]);
  T *data = cl::sycl::malloc_shared<T>(size, q);

  return shp::device_span<T>(data, size, rank);
}

// Return a range of spans, with one span allocated on each device using
// shared memory.
template <typename T>
std::vector<shp::device_span<T>> allocate_shared_spans(std::size_t size,
                                                       auto &&devices) {
  std::vector<shp::device_span<T>> spans;
  for (size_t rank = 0; rank < devices.size(); rank++) {
    spans.push_back(allocate_shared_span<T>(size, rank, devices));
  }
  return spans;
}

} // namespace shp
