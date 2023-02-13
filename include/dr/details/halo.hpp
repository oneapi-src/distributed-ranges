// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

namespace lib {

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
    drlog.debug(nostd::source_location::current(),
                "Halo constructed with {}/{} owned/halo\n", owned_groups.size(),
                halo_groups.size());
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
    drlog.debug(nostd::source_location::current(), "Halo exchange begin\n");
    receive(halo_groups_);
    send(owned_groups_);
  }

  /// Complete a halo exchange
  void exchange_finalize() {
    reduce_finalize(second);
    drlog.debug(nostd::source_location::current(), "Halo exchange finalize\n");
  }

  /// Begin a halo reduction
  void reduce_begin() {
    receive(owned_groups_);
    send(halo_groups_);
  }

  /// Complete a halo reduction
  void reduce_finalize(const auto &op) {
    for (int pending = requests_.size(); pending > 0; pending--) {
      int completed;
      MPI_Waitany(requests_.size(), requests_.data(), &completed,
                  MPI_STATUS_IGNORE);
      drlog.debug("Completed: {}\n", completed);
      auto &g = *map_[completed];
      if (g.receive && g.buffered) {
        g.unpack(op);
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

  struct multipllies_op {
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
      drlog.debug("Sending: {}\n", g.request_index);
      comm_.isend(g.data_pointer(), g.data_size(), g.rank(), g.tag(),
                  &requests_[g.request_index]);
    }
  }

  void receive(std::vector<Group> &receives) {
    for (auto &g : receives) {
      g.receive = true;
      drlog.debug("Receiving: {}\n", g.request_index);
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
    for (std::size_t i = 0; i < indices.size() - 1; i++) {
      buffered = buffered || (indices[i + 1] - indices[i] != 1);
    }
    indices_size_ = indices.size();
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
  communicator::tag tag_ = communicator::tag::halo_index;
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
  bool buffered = true;

  span_group(T *data, std::size_t size, std::size_t rank, communicator::tag tag,
             const Memory &memory)
      : data_(data, size), rank_(rank), tag_(tag), memory_(memory) {}

  span_group(std::span<T> data, std::size_t rank, communicator::tag tag)
      : data_(data), rank_(rank), tag_(tag) {}

  void unpack(const auto &op) {
    for (std::size_t i = 0; i < data_.size(); i++) {
      drlog.debug("unpack before {}, {}: {}\n", i, data_[i], *buffer);
      data_[i] = op(data_[i], *buffer++);
      drlog.debug("       after {}\n", data_[i]);
    }
  }

  void pack() { std::copy(data_.begin(), data_.end(), buffer); }
  std::size_t buffer_size() { return data_.size(); }

  std::size_t data_size() { return data_.size(); }
  T *data_pointer() { return buffer; }

  std::size_t rank() { return rank_; }

  auto tag() { return tag_; }

private:
  Memory memory_;
  std::span<T> data_;
  std::size_t rank_;
  communicator::tag tag_ = communicator::tag::invalid;
  ;
};

struct halo_bounds {
  /// Constructor
  halo_bounds(std::size_t radius = 0, bool per = false) {
    prev = radius;
    next = radius;
    periodic = per;
  }
  /// Constructor
  halo_bounds(std::size_t prv, std::size_t nxt, bool per = false) {
    prev = prv;
    next = nxt;
    periodic = per;
  }

  std::size_t prev, next;
  bool periodic;
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
                                  halo_groups(comm, {data, size}, hb)) {}

  span_halo(communicator comm, std::span<T> span, halo_bounds hb)
      : span_halo_impl<T, Memory>(comm, owned_groups(comm, span, hb),
                                  halo_groups(comm, span, hb)) {}

private:
  static std::vector<group_type>
  owned_groups(communicator comm, std::span<T> span, halo_bounds hb) {
    std::vector<group_type> owned;
    drlog.debug(nostd::source_location::current(),
                "owned groups {}/{} first/last\n", comm.first(), comm.last());
    if (hb.periodic || !comm.first()) {
      owned.emplace_back(span.subspan(hb.prev, hb.prev), comm.prev(),
                         communicator::tag::halo_reverse);
    }
    if (hb.periodic || !comm.last()) {
      owned.emplace_back(span.subspan(span.size() - 2 * hb.next, hb.next),
                         comm.next(), communicator::tag::halo_forward);
    }
    return owned;
  }

  static std::vector<group_type>
  halo_groups(communicator comm, std::span<T> span, halo_bounds hb) {
    std::vector<group_type> halo;
    if (hb.periodic || !comm.first()) {
      halo.emplace_back(span.first(hb.prev), comm.prev(),
                        communicator::tag::halo_forward);
    }
    if (hb.periodic || !comm.last()) {
      halo.emplace_back(span.last(hb.next), comm.next(),
                        communicator::tag::halo_reverse);
    }
    return halo;
  }
};

} // namespace lib
