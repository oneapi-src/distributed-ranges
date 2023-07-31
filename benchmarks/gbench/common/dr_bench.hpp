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

#include "cxxopts.hpp"
#include <benchmark/benchmark.h>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <vendor/source_location/source_location.hpp>

extern std::size_t comm_rank;
extern std::size_t ranks;

extern std::size_t default_vector_size;
extern std::size_t default_repetitions;

#define DR_BENCHMARK(x)                                                        \
  BENCHMARK(x)->UseRealTime()->Unit(benchmark::kMillisecond)

#ifdef SYCL_LANGUAGE_VERSION

inline auto device_info(sycl::device device) {
  return fmt::format("{}, max_compute_units: {}",
                     device.get_info<sycl::info::device::name>(),
                     device.get_info<sycl::info::device::max_compute_units>());
}

#endif

#ifdef BENCH_MHP

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

#endif

class Stats {
public:
  Stats(benchmark::State &state, std::size_t bytes_read = 0,
        std::size_t bytes_written = 0)
      : state_(state) {
    bytes_read_ = bytes_read;
    bytes_written_ = bytes_written;
  }

  ~Stats() {
    state_.SetBytesProcessed(reps_ * (bytes_read_ + bytes_written_));
    state_.counters["footprint"] = benchmark::Counter(
        (bytes_read_ + bytes_written_) / ranks, benchmark::Counter::kDefaults,
        benchmark::Counter::kIs1024);
  }

  void rep() { reps_++; }

private:
  benchmark::State &state_;
  std::size_t bytes_read_ = 0;
  std::size_t bytes_written_ = 0;
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

inline void add_configuration(int rank, std::string model, std::string runtime,
                              const cxxopts::ParseResult &options) {
  benchmark::AddCustomContext("default_vector_size",
                              std::to_string(default_vector_size));
  benchmark::AddCustomContext("default_repetitions",
                              std::to_string(default_repetitions));
  benchmark::AddCustomContext("rank", std::to_string(rank));
  benchmark::AddCustomContext("ranks", std::to_string(ranks));
  benchmark::AddCustomContext("model", model);
  benchmark::AddCustomContext("runtime", runtime);
  benchmark::AddCustomContext("lscpu", exec("lscpu"));
  if (options.count("context")) {
    for (std::string context :
         options["context"].as<std::vector<std::string>>()) {
      std::string delimeter = ":";
      auto split = context.find(delimeter);
      if (split == std::string::npos) {
        std::cerr << fmt::format("Context must use '{}' as delimiter: {}\n",
                                 delimeter, context);
        exit(1);
      }
      auto value_pos = split + delimeter.length();
      benchmark::AddCustomContext(
          context.substr(0, split),
          context.substr(value_pos, context.length() - value_pos));
    }
  }
}
