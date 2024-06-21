// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/mhp/global.hpp>

namespace dr::mhp::__detail {

template <typename T> void copy(const T *src, T *dst, std::size_t sz) {
  if (mhp::use_sycl()) {
#ifdef SYCL_LANGUAGE_VERSION
    sycl::queue().copy(src, dst, sz).wait();
#else
    assert(false);
#endif
  } else {
    memcpy(dst, src, sz * sizeof(T));
  }
}

template <typename T> class allocator {

public:
  T *allocate(std::size_t sz) {
    if (sz == 0) {
      return nullptr;
    }

    T *mem = nullptr;

    if (mhp::use_sycl()) {
#ifdef SYCL_LANGUAGE_VERSION
      mem = sycl::malloc<T>(sz, sycl_queue(), sycl_mem_kind());
#else
      assert(false);
#endif
    } else {
      mem = std_allocator_.allocate(sz);
    }

    assert(mem != nullptr);
    return mem;
  }

  void deallocate(T *ptr, std::size_t sz) {
    if (sz == 0) {
      assert(ptr == nullptr);
      return;
    }
    assert(ptr != nullptr);
#ifdef SYCL_LANGUAGE_VERSION
    if (mhp::use_sycl()) {
      sycl::free(ptr, sycl_queue());
      return;
    }
#endif

    std_allocator_.deallocate(ptr, sz);
  }

private:
  std::allocator<T> std_allocator_;
};

} // namespace dr::mhp::__detail
