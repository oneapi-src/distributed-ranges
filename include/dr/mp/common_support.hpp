// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// file for helper functions implemented for both SYCL and non-SYCL compilations

namespace dr::mp::__detail {

template <typename T> void copy(const T *src, T *dst, std::size_t sz) {
  if (mp::use_sycl()) {
    sycl_copy<T>(src, dst, sz);
  } else {
    memcpy(dst, src, sz * sizeof(T));
  }
}

} // namespace dr::mp::__detail
