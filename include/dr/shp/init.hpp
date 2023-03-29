// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <memory>
#include <span>
#include <sycl/sycl.hpp>
#include <type_traits>

#include <dr/shp/algorithms/execution_policy.hpp>

namespace shp {

namespace internal {

inline sycl::context *global_context_;

inline std::vector<sycl::device> devices_;

inline std::size_t ngpus_;

inline sycl::context global_context() { return *global_context_; }

inline std::size_t ngpus() { return ngpus_; }

inline std::span<sycl::device> global_devices() { return devices_; }

} // namespace internal

inline sycl::context context() { return internal::global_context(); }

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

  par_unseq = device_policy();
}

inline void finalize() { delete internal::global_context_; }

namespace __detail {

inline sycl::queue queue_for_rank(std::size_t rank) {
  assert(rank < internal::ngpus_);
  return sycl::queue(internal::global_context(),
                     internal::global_devices()[rank]);
}
inline sycl::queue default_queue() { return queue_for_rank(0); }

} // namespace __detail

} // namespace shp
