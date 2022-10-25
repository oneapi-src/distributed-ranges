#pragma once

#include <CL/sycl.hpp>
#include <memory>
#include <ranges>

namespace shp {

namespace internal {

inline sycl::context *global_context_;

inline sycl::context global_context() { return *global_context_; }

} // namespace internal

inline sycl::context context() { return internal::global_context(); }

template <typename R> inline void init(R &&devices) {
  internal::global_context_ = new sycl::context(std::forward<R>(devices));
}

} // namespace shp
