// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <CL/sycl.hpp>
#include <memory>
#include <ranges>
#include <span>
#include <type_traits>

#include <shp/algorithms/execution_policy.hpp>

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

template <std::ranges::range R>
inline void init(R &&devices)
  requires(std::is_same_v<cl::sycl::device,
                          std::remove_cvref_t<std::ranges::range_value_t<R>>>)
{
  internal::devices_.assign(std::ranges::begin(devices),
                            std::ranges::end(devices));
  internal::global_context_ = new sycl::context(internal::devices_);
  internal::ngpus_ = std::ranges::size(internal::devices_);

  par_unseq = device_policy(internal::devices_);
}

} // namespace shp
