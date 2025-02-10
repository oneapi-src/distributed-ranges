// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/mp/global.hpp>

namespace dr::mp::__detail {

template <typename T> class allocator {

public:
  T *allocate(std::size_t sz) {
    if (sz == 0) {
      return nullptr;
    }

    T *mem = nullptr;

    if (mp::use_sycl()) {
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
    std::cout << "deallocate(" << ptr << "," << sz << ")\n";
    if (sz == 0) {
      assert(ptr == nullptr);
      return;
    }
    assert(ptr != nullptr);
#ifdef SYCL_LANGUAGE_VERSION
    if (mp::use_sycl()) {
      std::cout << "deallocating with sycl\n";
      sycl::free(ptr, sycl_queue());
      return;
    }
#endif

    std::cout << "deallocating with std\n";
    std_allocator_.deallocate(ptr, sz);
  }

private:
  std::allocator<T> std_allocator_;
};

} // namespace dr::mp::__detail
