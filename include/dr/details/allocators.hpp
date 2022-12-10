namespace lib {

template <typename T>
class sycl_shared_allocator
    : public sycl::usm_allocator<T, sycl::usm::alloc::shared> {
private:
  using sycl_allocator_type = sycl::usm_allocator<T, sycl::usm::alloc::shared>;

public:
  sycl_shared_allocator(sycl::queue q = sycl::queue())
      : sycl_allocator_type(q), q_(q), policy_(q) {}

  const auto &policy() const { return policy_; }

private:
  sycl::queue q_;
  decltype(oneapi::dpl::execution::make_device_policy(sycl::queue{})) policy_;
};

} // namespace lib
