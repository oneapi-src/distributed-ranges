// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <memory>
#include <span>
#include <sycl/sycl.hpp>
#include <type_traits>

#include <dr/shp/algorithms/execution_policy.hpp>
#include <oneapi/dpl/execution>

namespace shp {

namespace internal {

inline sycl::context *global_context_;

inline std::vector<sycl::device> devices_;

inline std::vector<sycl::queue> queues_;

inline std::vector<oneapi::dpl::execution::device_policy<>> dpl_policies_;

inline std::size_t ngpus_;

inline sycl::context &global_context() { return *global_context_; }

inline std::size_t ngpus() { return ngpus_; }

inline std::span<sycl::device> global_devices() { return devices_; }

} // namespace internal

inline sycl::context &context() { return internal::global_context(); }

inline std::span<sycl::device> devices() { return internal::global_devices(); }

inline std::size_t nprocs() { return internal::ngpus(); }

inline device_policy par_unseq;

template <rng::range R>
inline void init(R &&devices)
  requires(
      std::is_same_v<sycl::device, std::remove_cvref_t<rng::range_value_t<R>>>)
{
  internal::devices_.assign(rng::begin(devices), rng::end(devices));
  internal::global_context_ = new sycl::context(internal::devices_);
  internal::ngpus_ = rng::size(internal::devices_);

  for (auto &&device : internal::devices_) {
    sycl::queue q(*internal::global_context_, device);
    internal::queues_.push_back(q);

    internal::dpl_policies_.emplace_back(internal::queues_.back());
  }

  par_unseq = device_policy(internal::devices_);
}

inline void finalize() {
  internal::dpl_policies_.clear();
  internal::queues_.clear();
  internal::devices_.clear();
  delete internal::global_context_;
}

namespace __detail {

inline auto default_queue() { return sycl::queue(); }

inline sycl::queue &queue(std::size_t rank) { return internal::queues_[rank]; }

inline auto &dpl_policy(std::size_t rank) {
  return internal::dpl_policies_[rank];
}

} // namespace __detail

} // namespace shp
