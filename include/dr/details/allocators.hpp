// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

namespace lib {

template <typename T>
class sycl_shared_allocator
    : public sycl::usm_allocator<T, sycl::usm::alloc::shared> {
private:
  using sycl_allocator_type = sycl::usm_allocator<T, sycl::usm::alloc::shared>;

public:
  sycl_shared_allocator(sycl::queue q = sycl::queue())
      : sycl_allocator_type(q), policy_(q) {}

  const auto &policy() const { return policy_; }

private:
  oneapi::dpl::execution::device_policy<> policy_;
};

} // namespace lib
