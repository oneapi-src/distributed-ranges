// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <unistd.h>

#include <dr/detail/sycl_utils.hpp>
#include <dr/mhp/sycl_support.hpp>

namespace dr::mhp {

namespace __detail {

struct global_context {
  void init() {
    void *data = nullptr;
    std::size_t size = 0;
    if (comm_.rank() == 0) {
      root_scratchpad_.resize(scratchpad_size_);
      data = root_scratchpad_.data();
      size = rng::size(root_scratchpad_) * sizeof(root_scratchpad_[0]);
    }
    root_win_.create(comm_, data, size);
    root_win_.fence();
  }

  ~global_context() {
    root_win_.fence();
    root_win_.free();
  }

  global_context() { init(); }
#ifdef SYCL_LANGUAGE_VERSION
  global_context(sycl::queue q)
      : sycl_queue_(q), dpl_policy_(q), use_sycl_(true) {
    init();
  }

  sycl::queue sycl_queue_;
  decltype(oneapi::dpl::execution::make_device_policy(
      std::declval<sycl::queue>())) dpl_policy_;
#endif

  static constexpr std::size_t scratchpad_size_ = 1000000;
  bool use_sycl_ = false;
  dr::communicator comm_;
  // container owns the window, we just track MPI handle
  std::set<MPI_Win> wins_;
  dr::rma_window root_win_;
  std::vector<char> root_scratchpad_;
};

inline global_context *global_context_ = nullptr;

inline bool finalized_ = false;
inline bool we_initialized_mpi_ = false;

inline auto gcontext() {
  assert(global_context_ && "Call mhp::init() after MPI_Init()");
  return global_context_;
}

// Initialize MPI if not already initialized.
inline void initialize_mpi() {
  int initialized;
  MPI_Initialized(&initialized);
  if (!initialized) {
    MPI_Init(nullptr, nullptr);
    we_initialized_mpi_ = true;
  }
}

// Finalize MPI *if* we initialized it and it has not been finalized.
inline void finalize_mpi() {
  int finalized;
  MPI_Finalized(&finalized);

  if (we_initialized_mpi_ && !finalized) {
    MPI_Finalize();
  }
}

} // namespace __detail

inline auto root_win() { return __detail::gcontext()->root_win_; }
inline dr::communicator &default_comm() { return __detail::gcontext()->comm_; }

inline bool finalized() { return __detail::finalized_; }
inline std::size_t rank() { return default_comm().rank(); }
inline std::size_t nprocs() { return default_comm().size(); } // dr-style ignore

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
  __detail::initialize_mpi();
  assert(__detail::global_context_ == nullptr &&
         "Do not call mhp::init() more than once");
  __detail::global_context_ = new __detail::global_context;
}

inline void finalize() {
  assert(__detail::global_context_ != nullptr);
  delete __detail::global_context_;
  __detail::global_context_ = nullptr;
  __detail::finalize_mpi();
  __detail::finalized_ = true;
}

inline std::string hostname() {
  constexpr std::size_t MH = 2048;
  char buf[MH + 1];
  gethostname(buf, MH);
  return std::string(buf);
}

#ifdef SYCL_LANGUAGE_VERSION
inline sycl::queue &sycl_queue() { return __detail::gcontext()->sycl_queue_; }
inline auto dpl_policy() { return __detail::gcontext()->dpl_policy_; }

inline sycl::queue select_queue(bool check_different_devices = false) {
  std::vector<sycl::device> devices;

  auto root_devices = sycl::platform().get_devices();

  for (auto &&root_device : root_devices) {
    dr::drlog.debug("Root device: {}\n",
                    root_device.get_info<sycl::info::device::name>());
    if (dr::__detail::partitionable(root_device)) {
      auto subdevices = root_device.create_sub_devices<
          sycl::info::partition_property::partition_by_affinity_domain>(
          sycl::info::partition_affinity_domain::numa);
      assert(rng::size(subdevices) > 0);

      for (auto &&subdevice : subdevices) {
        dr::drlog.debug("  add subdevice: {}\n",
                        subdevice.get_info<sycl::info::device::name>());
        devices.push_back(subdevice);
      }
    } else {
      dr::drlog.debug("  add root device: {}\n",
                      root_device.get_info<sycl::info::device::name>());
      devices.push_back(root_device);
    }
  }

  assert(rng::size(devices) > 0);
  const auto my_rank = dr::communicator(MPI_COMM_WORLD).rank();
  assert(!check_different_devices || my_rank < rng::size(devices));

  // Round robin assignment of devices to ranks
  return sycl::queue(devices[my_rank % rng::size(devices)]);
}

inline void init(sycl::queue q) {
  __detail::initialize_mpi();
  assert(__detail::global_context_ == nullptr &&
         "Do not call mhp::init() more than once");
  __detail::global_context_ = new __detail::global_context(q);
}

template <typename Selector = decltype(sycl::default_selector_v)>
inline void init(Selector &&selector = sycl::default_selector_v) {
  __detail::initialize_mpi();
  sycl::queue q = mhp::select_queue();
  init(q);
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
