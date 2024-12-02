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

template <typename Group> class halo_impl {
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
    for (auto &g : owned_groups_) {
      buffer_index.push_back(buffer_size_);
      g.request_index = i++;
      buffer_size_ += g.buffer_size();
      map_.push_back(&g);
    }
    for (auto &g : halo_groups_) {
      buffer_index.push_back(buffer_size_);
      g.request_index = i++;
      buffer_size_ += g.buffer_size();
      map_.push_back(&g);
    }
    buffer_ = memory_.allocate(buffer_size_);
    assert(buffer_ != nullptr);
    i = 0;
    for (auto &g : owned_groups_) {
      g.buffer = &buffer_[buffer_index[i++]];
    }
    for (auto &g : halo_groups_) {
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
    for (auto &g : sends) {
      g.pack();
      g.receive = false;
      DRLOG("sending: {}", g.request_index);
      comm_.isend(g.data_pointer(), g.data_size(), g.rank(), g.tag(),
                  &requests_[g.request_index]);
    }
  }

  void receive(std::vector<Group> &receives) {
    for (auto &g : receives) {
      g.receive = true;
      DRLOG("receiving: {}", g.request_index);
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

template <typename T, typename Memory = default_memory<T>> class index_group {
public:
  using element_type = T;
  using memory_type = Memory;
  T *buffer = nullptr;
  std::size_t request_index;
  bool receive;
  bool buffered;

  /// Constructor
  index_group(T *data, std::size_t rank,
              const std::vector<std::size_t> &indices, const Memory &memory)
      : memory_(memory), data_(data), rank_(rank) {
    buffered = false;
    for (std::size_t i = 0; i < rng::size(indices) - 1; i++) {
      buffered = buffered || (indices[i + 1] - indices[i] != 1);
    }
    indices_size_ = rng::size(indices);
    indices_ = memory_.template allocate<std::size_t>(indices_size_);
    assert(indices_ != nullptr);
    memory_.memcpy(indices_, indices.data(),
                   indices_size_ * sizeof(std::size_t));
  }

  index_group(const index_group &o)
      : buffer(o.buffer), request_index(o.request_index), receive(o.receive),
        buffered(o.buffered), memory_(o.memory_), data_(o.data_),
        rank_(o.rank_), indices_size_(o.indices_size_), tag_(o.tag_) {
    indices_ = memory_.template allocate<std::size_t>(indices_size_);
    assert(indices_ != nullptr);
    memory_.memcpy(indices_, o.indices_, indices_size_ * sizeof(std::size_t));
  }

  void unpack(const auto &op) {
    T *dpt = data_;
    auto n = indices_size_;
    auto *ipt = indices_;
    auto *b = buffer;
    memory_.offload([=]() {
      for (std::size_t i = 0; i < n; i++) {
        dpt[ipt[i]] = op(dpt[ipt[i]], b[i]);
      }
    });
  }

  void pack() {
    T *dpt = data_;
    auto n = indices_size_;
    auto *ipt = indices_;
    auto *b = buffer;
    memory_.offload([=]() {
      for (std::size_t i = 0; i < n; i++) {
        b[i] = dpt[ipt[i]];
      }
    });
  }

  std::size_t buffer_size() {
    if (buffered) {
      return indices_size_;
    }
    return 0;
  }

  T *data_pointer() {
    if (buffered) {
      return buffer;
    } else {
      return &data_[indices_[0]];
    }
  }

  std::size_t data_size() { return indices_size_; }

  std::size_t rank() { return rank_; }
  auto tag() { return tag_; }

  ~index_group() {
    if (indices_) {
      memory_.template deallocate<std::size_t>(indices_, indices_size_);
      indices_ = nullptr;
    }
  }

private:
  Memory memory_;
  T *data_ = nullptr;
  std::size_t rank_;
  std::size_t indices_size_;
  std::size_t *indices_;
  halo_tag tag_ = halo_tag::index;
};

template <typename T, typename Memory>
using unstructured_halo_impl = halo_impl<index_group<T, Memory>>;

template <typename T, typename Memory = default_memory<T>>
class unstructured_halo : public unstructured_halo_impl<T, Memory> {
public:
  using group_type = index_group<T, Memory>;
  using index_map = std::pair<std::size_t, std::vector<std::size_t>>;

  ///
  /// Constructor
  ///
  unstructured_halo(communicator comm, T *data,
                    const std::vector<index_map> &owned,
                    const std::vector<index_map> &halo,
                    const Memory &memory = Memory())
      : unstructured_halo_impl<T, Memory>(
            comm, make_groups(comm, data, owned, memory),
            make_groups(comm, data, halo, memory), memory) {}

private:
  static std::vector<group_type> make_groups(communicator comm, T *data,
                                             const std::vector<index_map> &map,
                                             const Memory &memory) {
    std::vector<group_type> groups;
    for (auto const &[rank, indices] : map) {
      groups.emplace_back(data, rank, indices, memory);
    }
    return groups;
  }
};

template <typename T, typename Memory = default_memory<T>> class span_group {
public:
  using element_type = T;
  using memory_type = Memory;
  T *buffer = nullptr;
  std::size_t request_index = 0;
  bool receive = false;
  bool buffered = false;

  span_group(std::span<T> data, std::size_t rank, halo_tag tag)
      : data_(data), rank_(rank), tag_(tag) {
#ifdef SYCL_LANGUAGE_VERSION
    if (use_sycl() && sycl_mem_kind() == sycl::usm::alloc::shared) {
      buffered = true;
    }
#endif
  }

  void unpack() {
    if (buffered) {
      if (mp::use_sycl()) {
        __detail::sycl_copy(buffer, buffer + rng::size(data_), data_.data());
      } else {
        std::copy(buffer, buffer + rng::size(data_), data_.data());
      }
    }
  }

  void pack() {
    if (buffered) {
      if (mp::use_sycl()) {
        __detail::sycl_copy(data_.data(), data_.data() + rng::size(data_),
                            buffer);
      } else {
        std::copy(data_.begin(), data_.end(), buffer);
      }
    }
  }
  std::size_t buffer_size() { return rng::size(data_); }

  std::size_t data_size() { return rng::size(data_); }

  T *data_pointer() {
    if (buffered) {
      return buffer;
    } else {
      return data_.data();
    }
  }

  std::size_t rank() { return rank_; }

  auto tag() { return tag_; }

private:
  Memory memory_;
  std::span<T> data_;
  std::size_t rank_;
  halo_tag tag_ = halo_tag::invalid;
};

struct halo_bounds {
  std::size_t prev = 0, next = 0;
  bool periodic = false;
};

template <typename T, typename Memory>
using span_halo_impl = halo_impl<span_group<T, Memory>>;

template <typename T, typename Memory = default_memory<T>>
class span_halo : public span_halo_impl<T, Memory> {
public:
  using group_type = span_group<T, Memory>;

  span_halo() : span_halo_impl<T, Memory>(communicator(), {}, {}) {}

  span_halo(communicator comm, T *data, std::size_t size, halo_bounds hb)
      : span_halo_impl<T, Memory>(comm, owned_groups(comm, {data, size}, hb),
                                  halo_groups(comm, {data, size}, hb)) {
    check(size, hb);
  }

  span_halo(communicator comm, std::span<T> span, halo_bounds hb)
      : span_halo_impl<T, Memory>(comm, owned_groups(comm, span, hb),
                                  halo_groups(comm, span, hb)) {}

private:
  void check(auto size, auto hb) {
    assert(size >= hb.prev + hb.next + std::max(hb.prev, hb.next));
  }

  static std::vector<group_type>
  owned_groups(communicator comm, std::span<T> span, halo_bounds hb) {
    std::vector<group_type> owned;
    DRLOG("owned groups {}/{} first/last", comm.first(), comm.last());
    if (hb.next > 0 && (hb.periodic || !comm.first())) {
      owned.emplace_back(span.subspan(hb.prev, hb.next), comm.prev(),
                         halo_tag::reverse);
    }
    if (hb.prev > 0 && (hb.periodic || !comm.last())) {
      owned.emplace_back(
          span.subspan(rng::size(span) - (hb.prev + hb.next), hb.prev),
          comm.next(), halo_tag::forward);
    }
    return owned;
  }

  static std::vector<group_type>
  halo_groups(communicator comm, std::span<T> span, halo_bounds hb) {
    std::vector<group_type> halo;
    if (hb.prev > 0 && (hb.periodic || !comm.first())) {
      halo.emplace_back(span.first(hb.prev), comm.prev(), halo_tag::forward);
    }
    if (hb.next > 0 && (hb.periodic || !comm.last())) {
      halo.emplace_back(span.last(hb.next), comm.next(), halo_tag::reverse);
    }
    return halo;
  }
};

template <typename T, typename Memory = default_memory<T>>
class cyclic_span_halo {
public:
  using group_type = span_group<T, Memory>;
  using halo_type = span_halo<T, Memory>;

  cyclic_span_halo(const std::vector<halo_type *>& halos)
    : halos_(halos) {
    for (const auto& halo : halos_) {
      assert(halo != nullptr);
    }
  }

  void exchange_begin() {
    halos_[next_comm_index_]->exchange_begin();
  }

  void exchange_finalize() {
    halos_[next_comm_index_]->exchange_finalize();
    //increment_index();
  }

  void exchange() {
    halos_[next_comm_index_]->exchange();
    //increment_index();
  }

  void reduce_begin() {
    halos_[next_comm_index_]->reduce_begin();
  }

  void reduce_finalize(const auto &op) {
    halos_[next_comm_index_]->reduce_finalize(op);
    //increment_index();
  }

  void reduce_finalize() {
    halos_[next_comm_index_]->reduce_finalize();
    //increment_index();
  }

  void swap() {
    increment_index();
  }

private:
  void increment_index() {
    next_comm_index_ = (next_comm_index_ + 1) % halos_.size();
  }

  std::vector<halo_type *> halos_;
  std::size_t next_comm_index_ = 0;
};

} // namespace dr::mp

#ifdef DR_FORMAT

template <>
struct fmt::formatter<dr::mp::halo_bounds> : formatter<string_view> {
  template <typename FmtContext>
  auto format(dr::mp::halo_bounds hb, FmtContext &ctx) {
    return fmt::format_to(ctx.out(), "prev: {} next: {}", hb.prev, hb.next);
  }
};

#endif
