// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

namespace dr::mhp {

#ifdef SYCL_LANGUAGE_VERSION

inline sycl::queue &sycl_queue();

namespace _detail {

template <typename T, std::size_t Alignment>
using shared_base_allocator =
    sycl::usm_allocator<T, sycl::usm::alloc::shared, Alignment>;

}; // namespace _detail

template <typename T, std::size_t Alignment = 0>
class sycl_shared_allocator
    : public _detail::shared_base_allocator<T, Alignment> {
public:
  sycl_shared_allocator(sycl::queue q = sycl_queue())
      : _detail::shared_base_allocator<T, Alignment>(q) {}
};

struct device_policy {
  device_policy(sycl::queue q = sycl_queue()) : queue(q), dpl_policy(q) {}

  sycl::queue queue;
  decltype(oneapi::dpl::execution::make_device_policy(queue)) dpl_policy;
};

#else // !SYCL_LANGUAGE_VERSION

struct device_policy {};

#endif // SYCL_LANGUAGE_VERSION

} // namespace dr::mhp
