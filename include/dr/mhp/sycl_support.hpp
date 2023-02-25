// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

namespace mhp {

namespace _detail {

template <typename T, std::size_t Alignment>
using shared_base_allocator =
    sycl::usm_allocator<T, sycl::usm::alloc::shared, Alignment>;
}; // namespace _detail

template <typename T, std::size_t Alignment = 0>
class sycl_shared_allocator
    : public _detail::shared_base_allocator<T, Alignment> {
public:
  sycl_shared_allocator(sycl::queue q)
      : _detail::shared_base_allocator<T, Alignment>(q) {}
  sycl_shared_allocator()
      : _detail::shared_base_allocator<T, Alignment>(sycl_queue()) {}
};

} // namespace mhp
