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
  global_context(sycl::queue q, sycl::usm::alloc kind)
      : sycl_queue_(q), sycl_mem_kind_(kind), dpl_policy_(q), use_sycl_(true) {
    init();
  }

  sycl::queue sycl_queue_;
  sycl::usm::alloc sycl_mem_kind_;
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
    DRLOG("initializing MPI");
    MPI_Init(nullptr, nullptr);
    we_initialized_mpi_ = true;
  } else {
    DRLOG("initializing MPI skipped - already initialized");
  }

#ifdef DRISHMEM
  DRLOG("calling ishmem_init()");
  ishmem_init();
#endif
}

// Finalize MPI *if* we initialized it and it has not been finalized.
inline void finalize_mpi() {
  int finalized;
  MPI_Finalized(&finalized);

  if (we_initialized_mpi_ && !finalized) {
    MPI_Finalize();
  }

#ifdef DRISHMEM
  DRLOG("calling ishmem_finalize()");
  ishmem_finalize();
#endif
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
#ifdef DRISHMEM
  DRLOG("global fence in ISHMEM");
  ishmem_fence();
  DRLOG("global fence in ISHMEM finished");
#endif
  for (auto win : __detail::gcontext()->wins_) {
    DRLOG("global fence, for window:{}", win);
    MPI_Win_fence(0, win);
  }
  DRLOG("global fence finished");
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
inline auto sycl_mem_kind() { return __detail::gcontext()->sycl_mem_kind_; }
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

inline void init(sycl::queue q,
                 sycl::usm::alloc kind = sycl::usm::alloc::shared) {
  __detail::initialize_mpi();
  assert(__detail::global_context_ == nullptr &&
         "Do not call mhp::init() more than once");
  __detail::global_context_ = new __detail::global_context(q, kind);
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

} // namespace dr::mhp

namespace dr::mhp::__detail {

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
