// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <memory>
#include <span>
#include <sycl/sycl.hpp>
#include <type_traits>
#include <vector>

#include <dr/shp/algorithms/execution_policy.hpp>
#include <oneapi/dpl/execution>

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
inline void init(R &&devices)
  requires(
      std::is_same_v<sycl::device, std::remove_cvref_t<rng::range_value_t<R>>>)
{
  __detail::devices_.assign(rng::begin(devices), rng::end(devices));
  __detail::global_context_ = new sycl::context(__detail::devices_);
  __detail::ngpus_ = rng::size(__detail::devices_);

  for (auto &&device : __detail::devices_) {
    sycl::queue q(*__detail::global_context_, device);
    __detail::queues_.push_back(q);

    __detail::dpl_policies_.emplace_back(__detail::queues_.back());
  }

  par_unseq = device_policy(__detail::devices_);
}

inline void finalize() {
  __detail::dpl_policies_.clear();
  __detail::queues_.clear();
  __detail::devices_.clear();
  delete __detail::global_context_;
}

namespace __detail {

inline sycl::queue &queue(std::size_t rank) { return queues_[rank]; }

inline sycl::queue &default_queue() { return queue(0); }

inline auto &dpl_policy(std::size_t rank) {
  return __detail::dpl_policies_[rank];
}

} // namespace __detail

} // namespace dr::shp
