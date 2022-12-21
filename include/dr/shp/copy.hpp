// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "device_ptr.hpp"
#include <CL/sycl.hpp>
#include <type_traits>

namespace shp {

template <typename T>
  requires(!std::is_const_v<T>)
cl::sycl::event
    copy_async(const T *first, const T *last, device_ptr<T> d_first) {
  cl::sycl::queue q;
  return q.memcpy(d_first.get_raw_pointer(), first, sizeof(T) * (last - first));
}

template <typename T>
  requires(!std::is_const_v<T>)
device_ptr<T> copy(const T *first, const T *last, device_ptr<T> d_first) {
  copy_async(first, last, d_first).wait();
  return d_first + (last - first);
}

template <typename T>
  requires(!std::is_const_v<T>)
cl::sycl::event copy_async(device_ptr<const T> first, device_ptr<const T> last,
                           T *d_first) {
  cl::sycl::queue q;
  return q.memcpy(d_first, first.get_raw_pointer(), last.get_raw_pointer());
}

template <typename T>
  requires(!std::is_const_v<T>)
T *copy(device_ptr<const T> first, device_ptr<const T> last, T *d_first) {
  copy_async(first, last, d_first).wait();
  return d_first + (last - first);
}

template <typename T>
  requires(!std::is_const_v<T>)
cl::sycl::event
    fill_async(device_ptr<T> first, device_ptr<T> last, const T &value) {
  cl::sycl::queue q;
  return q.fill(first.get_raw_pointer(), value, last - first);
}

template <typename T>
  requires(!std::is_const_v<T>)
void fill(device_ptr<T> first, device_ptr<T> last, const T &value) {
  cl::sycl::queue q;
  q.fill(first.get_raw_pointer(), value, last - first).wait();
}

} // namespace shp
