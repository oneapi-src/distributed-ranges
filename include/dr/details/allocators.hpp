namespace lib {

#ifdef SYCL_LANGUAGE_VERSION
template <typename T>
using shared_allocator = sycl::usm_allocator<T, sycl::usm::alloc::shared>;

#endif

} // namespace lib
