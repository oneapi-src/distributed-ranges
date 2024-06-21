// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// file for helper functions implemented for both SYCL and non-SYCL compilations

#include <dr/mhp/global.hpp>
#include <dr/mhp/sycl_support.hpp>

namespace dr::mhp::__detail {

template <typename T> void copy(const T *src, T *dst, std::size_t sz) {
  if (mhp::use_sycl()) {
    sycl_copy(src, dst, sz);
  } else {
    memcpy(dst, src, sz * sizeof(T));
  }
}

} // namespace dr::mhp::__detail
