// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <CL/sycl.hpp>
#include <dr/details/ranges_shim.hpp>
#include <span>
#include <vector>

namespace shp {

struct device_policy {
  device_policy(cl::sycl::device device) : devices_({device}) {}
  device_policy(cl::sycl::queue queue) : devices_({queue.get_device()}) {}

  device_policy() : devices_({cl::sycl::queue{}.get_device()}) {}

  template <rng::range R>
    requires(std::is_same_v<rng::range_value_t<R>, cl::sycl::device>)
  device_policy(R &&devices)
      : devices_(rng::begin(std::forward<R>(devices)),
                 rng::end(std::forward<R>(devices))) {}

  std::span<cl::sycl::device> get_devices() noexcept { return devices_; }

  std::span<const cl::sycl::device> get_devices() const noexcept {
    return devices_;
  }

private:
  std::vector<cl::sycl::device> devices_;
};

} // namespace shp
