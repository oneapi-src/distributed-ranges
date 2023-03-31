// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <type_traits>

#include <sycl/sycl.hpp>

#include <dr/shp/device_ptr.hpp>

namespace shp {

template <typename T>
using shared_allocator = sycl::usm_allocator<T, sycl::usm::alloc::shared>;

template <typename T, std::size_t Alignment = 0>
  requires(std::is_trivially_copyable_v<T>)
class device_allocator {
public:
  using value_type = T;
  using pointer = device_ptr<T>;
  using const_pointer = device_ptr<T>;
  using reference = device_ref<T>;
  using const_reference = device_ref<const T>;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  template <typename U>
  device_allocator(const device_allocator<U, Alignment> &other) noexcept
      : device_(other.get_device()), context_(other.get_context()) {}

  device_allocator(const sycl::queue &q) noexcept
      : device_(q.get_device()), context_(q.get_context()) {}
  device_allocator(const sycl::context &ctxt, const sycl::device &dev) noexcept
      : device_(dev), context_(ctxt) {}

  device_allocator(const device_allocator &) = default;
  device_allocator &operator=(const device_allocator &) = default;
  ~device_allocator() = default;

  using is_always_equal = std::false_type;

  pointer allocate(std::size_t size) {
    if constexpr (Alignment == 0) {
      return pointer(sycl::malloc_device<T>(size, device_, context_));
    } else {
      return pointer(
          sycl::aligned_alloc_device<T>(Alignment, size, device_, context_));
    }
  }

  void deallocate(pointer ptr, std::size_t n) {
    sycl::free(ptr.get_raw_pointer(), context_);
  }

  bool operator==(const device_allocator &) const = default;
  bool operator!=(const device_allocator &) const = default;

  template <typename U> struct rebind {
    using other = device_allocator<U, Alignment>;
  };

  sycl::device get_device() const noexcept { return device_; }

  sycl::context get_context() const noexcept { return context_; }

private:
  sycl::device device_;
  sycl::context context_;
};

} // namespace shp
