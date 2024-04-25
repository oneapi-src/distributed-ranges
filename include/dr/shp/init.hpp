// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cassert>
#include <memory>
#include <span>
#include <sycl/sycl.hpp>
#include <type_traits>
#include <vector>

#include <dr/shp/algorithms/execution_policy.hpp>
#include <dr/shp/util.hpp>
#include <oneapi/dpl/execution>

#include <fmt/ranges.h>

namespace dr::shp {

namespace __detail {

inline sycl::context *global_context_;

inline std::vector<sycl::device> devices_;

inline std::vector<sycl::queue> queues_;

inline std::vector<oneapi::dpl::execution::device_policy<>> dpl_policies_;

inline std::size_t ngpus_;

inline sycl::context &global_context() { return *global_context_; }

inline std::size_t ngpus() { return ngpus_; }

inline std::span<sycl::device> global_devices() { return devices_; }

} // namespace __detail

inline sycl::context &context() { return __detail::global_context(); }

inline std::span<sycl::device> devices() { return __detail::global_devices(); }

inline std::size_t nprocs() { return __detail::ngpus(); }

inline device_policy par_unseq;

template <rng::range R>
inline void init(sycl::context context, R &&devices)
  requires(
      std::is_same_v<sycl::device, std::remove_cvref_t<rng::range_value_t<R>>>)
{
  __detail::devices_.assign(rng::begin(devices), rng::end(devices));
  __detail::global_context_ = new sycl::context(context);
  __detail::ngpus_ = rng::size(__detail::devices_);

  for (auto &&device : __detail::devices_) {
    sycl::queue q(*__detail::global_context_, device);
    __detail::queues_.push_back(q);

    __detail::dpl_policies_.emplace_back(__detail::queues_.back());
  }

  par_unseq = device_policy(__detail::devices_);
}

template <rng::range R>
inline void init(R &&devices)
  requires(
      std::is_same_v<sycl::device, std::remove_cvref_t<rng::range_value_t<R>>>)
{
  sycl::context context(devices);
  init(context, devices);
}

void exception_handler(sycl::exception_list exceptions) {
  for (const std::exception_ptr& e : exceptions) {
    try {
      std::rethrow_exception(e);
    } catch(const sycl::exception& e) {
      if (e.code().value() != 34) {
        throw;
      }
    /*
      std::string exception_string(e.what());
      if (exception_string.find("PI_ERROR_INVALID_CONTEXT") == std::string::npos) {
      }
      */
    }
  }
}

template <__detail::sycl_device_selector Selector>
inline void init_cuda(Selector&& selector) {
  std::vector<sycl::device> devices;

  for (auto&& platform : sycl::platform::get_platforms()) {
    std::cout << "Platform: " << platform.get_info<sycl::info::platform::name>() << std::endl;

    if (platform.get_backend() == sycl::backend::ext_oneapi_cuda) {
      for (auto&& device : platform.get_devices()) {
        std::cout << "  Device: " << device.get_info<sycl::info::device::name>()
                  << std::endl;
        devices.push_back(device);
      }
    }
  }

  __detail::devices_.assign(rng::begin(devices), rng::end(devices));
  __detail::global_context_ = new sycl::context(devices[0]);
  __detail::ngpus_ = rng::size(__detail::devices_);

  for (auto &&device : __detail::devices_) {
    // sycl::queue q(device, exception_handler);
    sycl::queue q(device);
    __detail::queues_.push_back(q);

    __detail::dpl_policies_.emplace_back(__detail::queues_.back());
  }
}

template <__detail::sycl_device_selector Selector>
inline void init(Selector &&selector) {
  sycl::platform p(selector);

  if (p.get_backend() == sycl::backend::ext_oneapi_cuda) {
    init_cuda(selector);
  } else {
    auto devices = get_numa_devices(selector);
    init(devices);
  }
}

inline void init() { init(sycl::default_selector_v); }

inline void finalize() {
  fmt::print("Destroying policies...\n");
  __detail::dpl_policies_.clear();
  fmt::print("Queues...\n");
  __detail::queues_.clear();
  fmt::print("Devices...\n");
  __detail::devices_.clear();
  fmt::print("Context...\n");
  delete __detail::global_context_;
}

inline void check_queues() {
  for (auto&& queue : __detail::queues_) {
    queue.wait_and_throw();
  }

  /*
  for (auto&& policy : __detail::dpl_policies_) {
    fmt::print("Looking at policy...\n");
    try {
      policy.queue().wait_and_throw();
    } catch (...) {
      fmt::print("Caught exception\n");
    }
  }
  */
}

namespace __detail {

inline sycl::queue &queue(std::size_t rank) { return queues_[rank]; }

// Retrieve global queues because of CMPLRLLVM-47008
inline sycl::queue &queue(const sycl::device &device) {
  for (std::size_t rank = 0; rank < shp::nprocs(); rank++) {
    if (shp::devices()[rank] == device) {
      return queue(rank);
    }
  }
  assert(false);
  // Reaches here with -DNDEBUG
  return queue(0);
}

inline sycl::queue &default_queue() { return queue(0); }

inline auto &dpl_policy(std::size_t rank) {
  return __detail::dpl_policies_[rank];
}

} // namespace __detail

} // namespace dr::shp
