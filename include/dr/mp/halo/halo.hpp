// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/mp/global.hpp>
#include <dr/mp/sycl_support.hpp>

namespace dr::mp {

  enum class halo_tag {
    invalid,
    forward,
    reverse,
    index,
  };

  struct halo_bounds {
    // How many values before and after the data segment are in halo
    std::size_t prev = 0, next = 0;
    bool periodic = false;
  };

  template<typename Group>
  class halo_impl {
    using T = typename Group::element_type;
    using Memory = typename Group::memory_type;

  public:
    using group_type = Group;

    // Destructor frees buffer_, so cannot copy
    halo_impl(const halo_impl &) = delete;

    halo_impl operator=(const halo_impl &) = delete;

    /// halo constructor
    halo_impl(communicator comm, const std::vector<Group> &owned_groups,
         const std::vector<Group> &halo_groups,
         const Memory &memory = Memory())
        : comm_(comm), halo_groups_(halo_groups), owned_groups_(owned_groups),
          memory_(memory) {
      DRLOG("Halo constructed with {}/{} owned/halo", rng::size(owned_groups),
            rng::size(halo_groups));
      buffer_size_ = 0;
      std::size_t i = 0;
      std::vector<std::size_t> buffer_index;
      for (auto &g: owned_groups_) {
        buffer_index.push_back(buffer_size_);
        g.request_index = i++;
        buffer_size_ += g.buffer_size();
        map_.push_back(&g);
      }
      for (auto &g: halo_groups_) {
        buffer_index.push_back(buffer_size_);
        g.request_index = i++;
        buffer_size_ += g.buffer_size();
        map_.push_back(&g);
      }
      buffer_ = memory_.allocate(buffer_size_);
      assert(buffer_ != nullptr);
      i = 0;
      for (auto &g: owned_groups_) {
        g.buffer = &buffer_[buffer_index[i++]];
      }
      for (auto &g: halo_groups_) {
        g.buffer = &buffer_[buffer_index[i++]];
      }
      requests_.resize(i);
    }

    /// Begin a halo exchange
    void exchange_begin() {
      DRLOG("Halo exchange receiving");
      receive(halo_groups_);
      DRLOG("Halo exchange sending");
      send(owned_groups_);
      DRLOG("Halo exchange begin finished");
    }

    /// Complete a halo exchange
    void exchange_finalize() {
      DRLOG("Halo exchange finalize started");
      reduce_finalize();
      DRLOG("Halo exchange finalize finished");
    }

    void exchange() {
      exchange_begin();
      exchange_finalize();
    }

    /// Begin a halo reduction
    void reduce_begin() {
      receive(owned_groups_);
      send(halo_groups_);
    }

    /// Complete a halo reduction
    void reduce_finalize(const auto &op) {
      for (int pending = rng::size(requests_); pending > 0; pending--) {
        int completed;
        MPI_Waitany(rng::size(requests_), requests_.data(), &completed,
                    MPI_STATUS_IGNORE);
        DRLOG("reduce_finalize(op) waitany completed: {}", completed);
        auto &g = *map_[completed];
        if (g.receive && g.buffered) {
          g.unpack(op);
        }
      }
    }

    /// Complete a halo reduction
    void reduce_finalize() {
      for (int pending = rng::size(requests_); pending > 0; pending--) {
        int completed;
        MPI_Waitany(rng::size(requests_), requests_.data(), &completed,
                    MPI_STATUS_IGNORE);
        DRLOG("reduce_finalize() waitany completed: {}", completed);
        auto &g = *map_[completed];
        if (g.receive && g.buffered) {
          g.unpack();
        }
      }
    }

    struct second_op {
      T operator()(T &a, T &b) const { return b; }
    } second;

    struct plus_op {
      T operator()(T &a, T &b) const { return a + b; }
    } plus;

    struct max_op {
      T operator()(T &a, T &b) const { return std::max(a, b); }
    } max;

    struct min_op {
      T operator()(T &a, T &b) const { return std::min(a, b); }
    } min;

    struct multiplies_op {
      T operator()(T &a, T &b) const { return a * b; }
    } multiplies;

    ~halo_impl() {
      if (buffer_) {
        memory_.deallocate(buffer_, buffer_size_);
        buffer_ = nullptr;
      }
    }

  private:
    void send(std::vector<Group> &sends) {
      for (auto &g: sends) {
        g.pack();
        g.receive = false;
        DRLOG("sending: {}", g.request_index);
//        std::cout << "send(" << g.data_pointer() << ", " << g.data_size() << ", " << g.rank() << ", <tag>, " << &requests_[g.request_index] << ")\n";
        comm_.isend(g.data_pointer(), g.data_size(), g.rank(), g.tag(),
                    &requests_[g.request_index]);
      }
    }

    void receive(std::vector<Group> &receives) {
      for (auto &g: receives) {
        g.receive = true;
        DRLOG("receiving: {}", g.request_index);
//        std::cout << "recv(" << g.data_pointer() << ", " << g.data_size() << ", " << g.rank() << ", <tag>, " << &requests_[g.request_index] << ")\n";
        comm_.irecv(g.data_pointer(), g.data_size(), g.rank(), g.tag(),
                    &requests_[g.request_index]);
      }
    }

    communicator comm_;
    std::vector<Group> halo_groups_, owned_groups_;
    T *buffer_ = nullptr;
    std::size_t buffer_size_;
    std::vector<MPI_Request> requests_;
    std::vector<Group *> map_;
    Memory memory_;
  };

  template <typename T, typename... Ts>
  void halo_exchange(auto&& f, T &dv, Ts &...dvs) {
    for (std::size_t step = 0; step < dv.dist().redundancy(); step++) {
      f(dv, dvs...);
    }
    halo(dv).exchange();
  }
}
