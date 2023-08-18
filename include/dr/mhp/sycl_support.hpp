// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#ifdef SYCL_LANGUAGE_VERSION

namespace dr::mhp {

sycl::queue &sycl_queue();

} // namespace dr::mhp

namespace dr::mhp::__detail {

template <typename T> T sycl_get(T &v) {
  T temp;
  sycl_queue().memcpy(&temp, &v, sizeof(v)).wait();
  return temp;
}

template <typename T> void sycl_copy(T *begin, T *end, T *dst) {
  sycl_queue().memcpy(dst, begin, (end - begin) * sizeof(T)).wait();
}

template <typename T, std::size_t Alignment>
using shared_base_allocator =
    sycl::usm_allocator<T, sycl::usm::alloc::shared, Alignment>;

}; // namespace dr::mhp::__detail

namespace dr::mhp {

template <typename T, std::size_t Alignment = 0>
class sycl_shared_allocator
    : public __detail::shared_base_allocator<T, Alignment> {
public:
  sycl_shared_allocator(sycl::queue q = sycl_queue())
      : __detail::shared_base_allocator<T, Alignment>(q) {}
};

struct device_policy {
  device_policy(sycl::queue q = sycl_queue()) : queue(q), dpl_policy(q) {}

  sycl::queue queue;
  decltype(oneapi::dpl::execution::make_device_policy(queue)) dpl_policy;
};

} // namespace dr::mhp

#else // !SYCL_LANGUAGE_VERSION

namespace dr::mhp {

struct device_policy {};

} // namespace dr::mhp

namespace dr::mhp::__detail {

// define here to avoid ifdefs where it is called
template <typename T> T sycl_get(T &v) {
  assert(false);
  return v;
}

} // namespace dr::mhp::__detail

#endif // SYCL_LANGUAGE_VERSION
