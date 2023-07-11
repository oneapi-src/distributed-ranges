// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/mhp/sycl_support.hpp>

namespace dr::mhp {

namespace __detail {

struct global_context {
  global_context() {}
#ifdef SYCL_LANGUAGE_VERSION
  global_context(sycl::queue q)
      : sycl_queue_(q), dpl_policy_(q), use_sycl_(true) {}
  sycl::queue sycl_queue_;
  decltype(oneapi::dpl::execution::make_device_policy(
      std::declval<sycl::queue>())) dpl_policy_;
#endif

  bool use_sycl_ = false;
  dr::communicator comm_;
  // container owns the window, we just track MPI handle
  std::set<MPI_Win> wins_;
};

inline global_context *global_context_ = nullptr;

inline auto gcontext() {
  assert(global_context_ && "Call mhp::init() after MPI_Init()");
  return global_context_;
}

} // namespace __detail

inline void final() {
  delete __detail::global_context_;
  __detail::global_context_ = nullptr;
}

inline dr::communicator &default_comm() { return __detail::gcontext()->comm_; }

inline std::set<MPI_Win> &active_wins() { return __detail::gcontext()->wins_; }

inline void barrier() { __detail::gcontext()->comm_.barrier(); }
inline auto use_sycl() { return __detail::gcontext()->use_sycl_; }

inline void fence() {
  dr::drlog.debug("fence\n");
  for (auto win : __detail::gcontext()->wins_) {
    MPI_Win_fence(0, win);
  }
}

inline void init() {
  assert(__detail::global_context_ == nullptr &&
         "Do not call mhp::init() more than once");
  __detail::global_context_ = new __detail::global_context;
}

#ifdef SYCL_LANGUAGE_VERSION
inline sycl::queue sycl_queue() { return __detail::gcontext()->sycl_queue_; }
inline auto dpl_policy() { return __detail::gcontext()->dpl_policy_; }

inline void init(sycl::queue q) {
  assert(__detail::global_context_ == nullptr &&
         "Do not call mhp::init() more than once");
  __detail::global_context_ = new __detail::global_context(q);
}

inline sycl::queue select_queue(MPI_Comm comm = MPI_COMM_WORLD) {
  std::vector<sycl::device> devices;

  auto root_devices = sycl::platform().get_devices();

  for (auto &&root_device : root_devices) {
    auto subdevices = root_device.create_sub_devices<
        sycl::info::partition_property::partition_by_affinity_domain>(
        sycl::info::partition_affinity_domain::numa);

    for (auto &&subdevice : subdevices) {
      devices.push_back(subdevice);
    }
  }

  assert(rng::size(devices) > 0);
  // Round robin assignment of devices to ranks
  return sycl::queue(
      devices[dr::communicator(comm).rank() % rng::size(devices)]);
}

#else // SYCL_LANGUAGE_VERSION
inline auto sycl_queue() {
  assert(false);
  return 0;
}
inline const auto &dpl_policy() {
  assert(false);
  return std::execution::seq;
}

#endif // SYCL_LANGUAGE_VERSION

template <typename T> class default_allocator {

  struct __dr_unique_ptr_deleter {
    std::size_t allocated_size;
    void operator()(T *ptr) {
      default_allocator<T>().deallocate(ptr, allocated_size);
    }
  };

public:
  default_allocator() {
#ifdef SYCL_LANGUAGE_VERSION
    if (mhp::use_sycl()) {
      sycl_allocator_ = sycl_shared_allocator<T>(sycl_queue());
    }
#endif
  }

  T *allocate(std::size_t sz) {
    if (sz == 0) {
      return nullptr;
    }
#ifdef SYCL_LANGUAGE_VERSION
    if (mhp::use_sycl()) {
      return sycl_allocator_.allocate(sz);
    }
#endif

    return std_allocator_.allocate(sz);
  }

  std::unique_ptr<T, __dr_unique_ptr_deleter> allocate_unique(std::size_t sz) {
    return std::unique_ptr<T, __dr_unique_ptr_deleter>(
        allocate(sz), __dr_unique_ptr_deleter{sz});
  }

  void deallocate(T *ptr, std::size_t sz) {
    if (sz == 0) {
      assert(ptr == nullptr);
      return;
    }
    assert(ptr != nullptr);
#ifdef SYCL_LANGUAGE_VERSION
    if (mhp::use_sycl()) {
      sycl_allocator_.deallocate(ptr, sz);
      return;
    }
#endif

    std_allocator_.deallocate(ptr, sz);
  }

private:
#ifdef SYCL_LANGUAGE_VERSION
  sycl_shared_allocator<T> sycl_allocator_;
#endif
  std::allocator<T> std_allocator_;
};

} // namespace dr::mhp
