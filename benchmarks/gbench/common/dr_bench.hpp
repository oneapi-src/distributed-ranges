// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <array>
#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#ifdef SYCL_LANGUAGE_VERSION
#include <sycl/sycl.hpp>
#endif

#include <benchmark/benchmark.h>
#include <cxxopts.hpp>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <vendor/source_location/source_location.hpp>

#include <dr/detail/logger.hpp>
#include <dr/detail/ranges_shim.hpp>
#include <dr/detail/sycl_utils.hpp>

extern std::size_t comm_rank;
extern std::size_t ranks;

extern std::size_t default_vector_size;
extern std::size_t default_repetitions;
extern bool weak_scaling;

#define DR_BENCHMARK(x) DR_BENCHMARK_BASE(x)->MinWarmUpTime(.1)->MinTime(.1)

#define DR_BENCHMARK_BASE(x)                                                   \
  BENCHMARK(x)->UseRealTime()->Unit(benchmark::kMillisecond)

#define DR_BENCHMARK_REGISTER_F(fixture, case)                                 \
  BENCHMARK_REGISTER_F(fixture, case)                                          \
      ->UseRealTime()                                                          \
      ->Unit(benchmark::kMillisecond)                                          \
      ->MinWarmUpTime(.1)                                                      \
      ->MinTime(.1)

#ifdef SYCL_LANGUAGE_VERSION
inline auto device_info(sycl::device device) {
  return fmt::format("{}, max_compute_units: {}",
                     device.get_info<sycl::info::device::name>(),
                     device.get_info<sycl::info::device::max_compute_units>());
}
#endif

#ifdef BENCH_MHP
#ifdef SYCL_LANGUAGE_VERSION

inline sycl::context *mhp_global_context_ = nullptr;
inline std::vector<sycl::device> devices;

inline sycl::queue get_queue() {
  if (mhp_global_context_ != nullptr) {
    return sycl::queue(*mhp_global_context_, devices[0]);
  }

  auto root_devices = sycl::platform().get_devices();

  for (auto &&[idx, root_device] : rng::views::enumerate(root_devices)) {
    dr::drlog.debug("Root device no {}: {}\n", idx,
                    root_device.get_info<sycl::info::device::name>());
    if (dr::__detail::partitionable(root_device)) {
      auto subdevices = root_device.create_sub_devices<
          sycl::info::partition_property::partition_by_affinity_domain>(
          sycl::info::partition_affinity_domain::numa);
      assert(rng::size(subdevices) > 0);

      for (auto &&subdevice : subdevices) {
        dr::drlog.debug("  add subdevice: {}\n",
                        subdevice.get_info<sycl::info::device::name>());
        devices.push_back(subdevice);
      }
    } else {
      dr::drlog.debug("  add root device: {}\n",
                      root_device.get_info<sycl::info::device::name>());
      devices.push_back(root_device);
    }
  }

  assert(rng::size(devices) > 0);

  mhp_global_context_ = new sycl::context(devices);
  return sycl::queue(*mhp_global_context_, devices[0]);
}

#endif

#include "dr/mhp.hpp"

namespace xhp = dr::mhp;

extern std::size_t stencil_steps;
extern std::size_t num_rows;
extern std::size_t num_columns;
extern bool check_results;

#endif

#ifdef BENCH_SHP

#include "dr/shp.hpp"

namespace xhp = dr::shp;

inline sycl::queue &get_queue() { return dr::shp::__detail::default_queue(); }

#endif

class Stats {
public:
  Stats(benchmark::State &state, std::size_t bytes_read = 0,
        std::size_t bytes_written = 0, std::size_t flops = 0)
      : state_(state) {
    bytes_read_ = bytes_read;
    bytes_written_ = bytes_written;
    flops_ = flops;
  }

  ~Stats() {
    if (flops_ > 0) {
      state_.counters["flops"] =
          benchmark::Counter(flops_, benchmark::Counter::kIsRate);
    }

    std::size_t mem = bytes_read_ + bytes_written_;
    if (mem > 0) {
      state_.SetBytesProcessed(reps_ * mem);
      state_.counters["footprint"] =
          benchmark::Counter(mem / ranks, benchmark::Counter::kDefaults,
                             benchmark::Counter::kIs1024);
    }
  }

  void rep() { reps_++; }

private:
  benchmark::State &state_;
  std::size_t bytes_read_ = 0;
  std::size_t bytes_written_ = 0;
  std::size_t flops_ = 0;
  std::size_t reps_ = 0;
};

inline std::string exec(const char *cmd) {
  std::array<char, 128> buffer;
  std::string result;
  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
  if (!pipe) {
    throw std::runtime_error("popen() failed!");
  }
  while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
    result += buffer.data();
  }
  return result;
}

inline void add_configuration(int rank, const cxxopts::ParseResult &options) {
  benchmark::AddCustomContext("hostname", exec("hostname"));
  benchmark::AddCustomContext("lscpu", exec("lscpu"));
  benchmark::AddCustomContext("numactl", exec("numactl -H"));
  benchmark::AddCustomContext("default_vector_size",
                              std::to_string(default_vector_size));
  benchmark::AddCustomContext("default_repetitions",
                              std::to_string(default_repetitions));
  benchmark::AddCustomContext("rank", std::to_string(rank));
  benchmark::AddCustomContext("ranks", std::to_string(ranks));
  benchmark::AddCustomContext("weak-scaling", std::to_string(weak_scaling));
  if (options.count("context")) {
    for (std::string context :
         options["context"].as<std::vector<std::string>>()) {
      std::string delimiter = ":";
      auto split = context.find(delimiter);
      if (split == std::string::npos) {
        std::cerr << fmt::format("Context must use '{}' as delimiter: {}\n",
                                 delimiter, context);
        exit(1);
      }
      auto value_pos = split + delimiter.length();
      benchmark::AddCustomContext(
          context.substr(0, split),
          context.substr(value_pos, context.length() - value_pos));
    }
  }
}
