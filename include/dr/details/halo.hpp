namespace lib {

template <typename Group> class halo_impl {
  using T = typename Group::element_type;

public:
  using group_type = Group;

  /// halo constructor
  halo_impl(communicator comm, const std::vector<Group> &owned_groups,
            const std::vector<Group> &halo_groups)
      : comm_(comm), halo_groups_(halo_groups), owned_groups_(owned_groups) {
    drlog.debug(nostd::source_location::current(),
                "Halo constructed with {}/{} owned/halo\n", owned_groups.size(),
                halo_groups.size());
    std::size_t buffer_size = 0;
    std::size_t i = 0;
    for (auto &g : owned_groups_) {
      g.buffer_index = buffer_size;
      g.request_index = i++;
      buffer_size += g.buffer_size();
      map_.push_back(&g);
    }
    for (auto &g : halo_groups_) {
      g.buffer_index = buffer_size;
      g.request_index = i++;
      buffer_size += g.buffer_size();
      map_.push_back(&g);
    }
    buffer_.resize(buffer_size);
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
      if (g.receive) {
        g.unpack(&buffer_[g.buffer_index], op);
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

private:
  void send(std::vector<Group> &sends) {
    for (auto &g : sends) {
      auto b = &buffer_[g.buffer_index];
      g.pack(b);
      g.receive = false;
      drlog.debug("Sending: {}\n", g.request_index);
      comm_.isend(b, g.buffer_size(), g.rank(), g.tag(),
                  &requests_[g.request_index]);
    }
  }

  void receive(std::vector<Group> &receives) {
    for (auto &g : receives) {
      g.receive = true;
      drlog.debug("Receiving: {}\n", g.request_index);
      comm_.irecv(&buffer_[g.buffer_index], g.buffer_size(), g.rank(), g.tag(),
                  &requests_[g.request_index]);
    }
  }

  communicator comm_;
  std::vector<Group> halo_groups_, owned_groups_;
  std::vector<T> buffer_;
  std::vector<MPI_Request> requests_;
  std::vector<Group *> map_;
};

template <typename T> class index_group {
public:
  using element_type = T;
  std::size_t buffer_index;
  std::size_t request_index;
  bool receive;

  /// Constructor
  index_group(T *data, std::size_t rank,
              const std::vector<std::size_t> &indices)
      : data_(data), rank_(rank), indices_(indices) {}

  void unpack(T *buffer, const auto &op) {
    for (auto i : indices_) {
      drlog.debug("unpack before {}, {}: {}\n", i, data_[i], *buffer);
      data_[i] = op(data_[i], *buffer++);
      drlog.debug("       after {}\n", data_[i]);
    }
  }

  void pack(T *buffer) {
    for (auto i : indices_) {
      drlog.debug("pack {}, {}\n", i, data_[i]);
      *buffer++ = data_[i];
    }
  }
  std::size_t buffer_size() { return indices_.size(); }

  std::size_t rank() { return rank_; }
  auto tag() { return tag_; }

private:
  T *data_ = nullptr;
  std::size_t rank_;
  std::vector<std::size_t> indices_;
  communicator::tag tag_ = communicator::tag::halo_index;
};

template <typename T> using unstructured_halo_impl = halo_impl<index_group<T>>;

template <typename T>
class unstructured_halo : public unstructured_halo_impl<T> {
public:
  using group_type = index_group<T>;
  using index_map = std::pair<std::size_t, std::vector<std::size_t>>;

  ///
  /// Constructor
  ///
  unstructured_halo(communicator comm, T *data,
                    const std::vector<index_map> &owned,
                    const std::vector<index_map> &halo)
      : unstructured_halo_impl<T>(comm, make_groups(comm, data, owned),
                                  make_groups(comm, data, halo)) {}

private:
  static std::vector<group_type>
  make_groups(communicator comm, T *data, const std::vector<index_map> &map) {
    std::vector<group_type> groups;
    for (auto const &[rank, indices] : map) {
      groups.emplace_back(data, rank, indices);
    }
    return groups;
  }
};

template <typename T> class span_group {
public:
  using element_type = T;
  std::size_t buffer_index;
  std::size_t request_index;
  bool receive;

  span_group(T *data, std::size_t size, std::size_t rank, communicator::tag tag)
      : data_(data, size), rank_(rank), tag_(tag) {}

  span_group(std::span<T> data, std::size_t rank, communicator::tag tag)
      : data_(data), rank_(rank), tag_(tag) {}

  void unpack(T *buffer, const auto &op) {
    for (std::size_t i = 0; i < data_.size(); i++) {
      drlog.debug("unpack before {}, {}: {}\n", i, data_[i], *buffer);
      data_[i] = op(data_[i], *buffer++);
      drlog.debug("       after {}\n", data_[i]);
    }
  }

  void pack(T *buffer) { std::copy(data_.begin(), data_.end(), buffer); }
  std::size_t buffer_size() { return data_.size(); }

  std::size_t rank() { return rank_; }

  auto tag() { return tag_; }

private:
  std::span<T> data_;
  std::size_t rank_;
  std::size_t radius_;
  communicator::tag tag_ = communicator::tag::invalid;
  ;
};

template <int Rank> class stencil {
public:
  struct dimension_type {
    std::size_t prev, next;
  };
  using radius_type = std::array<dimension_type, Rank>;
  /// Constructor
  stencil(bool periodic, std::size_t radius = 0) {
    radius_[0].prev = radius;
    radius_[0].next = radius;
    periodic_ = periodic;
    assert(Rank == 1);
  }

  /// Constructor
  stencil(std::size_t radius = 0) {
    radius_[0].prev = radius;
    radius_[0].next = radius;
    periodic_ = false;
    assert(Rank == 1);
  }

  /// Returns radius of stencil
  const radius_type &radius() const { return radius_; }

  /// Returns True if boundary is periodic (wraps around)
  bool periodic() const { return periodic_; }

private:
  radius_type radius_;
  bool periodic_;
};

template <typename T> using span_halo_impl = halo_impl<span_group<T>>;

template <typename T> class span_halo : public span_halo_impl<T> {
public:
  using group_type = span_group<T>;

  span_halo() : span_halo_impl<T>(communicator(), {}, {}) {}

  span_halo(communicator comm, T *data, std::size_t size, stencil<1> stencl)
      : span_halo_impl<T>(comm, owned_groups(comm, {data, size}, stencl),
                          halo_groups(comm, {data, size}, stencl)) {}

  span_halo(communicator comm, std::span<T> span, stencil<1> stencl)
      : span_halo_impl<T>(comm, owned_groups(comm, span, stencl),
                          halo_groups(comm, span, stencl)) {}

private:
  static std::vector<group_type>
  owned_groups(communicator comm, std::span<T> span, stencil<1> stencl) {
    const auto &radius = stencl.radius()[0];
    std::vector<group_type> owned;
    drlog.debug(nostd::source_location::current(),
                "owned groups {}/{} first/last\n", comm.first(), comm.last());
    if (stencl.periodic() || !comm.first()) {
      owned.emplace_back(span.subspan(radius.prev, radius.prev), comm.prev(),
                         communicator::tag::halo_reverse);
    }
    if (stencl.periodic() || !comm.last()) {
      owned.emplace_back(
          span.subspan(span.size() - 2 * radius.next, radius.next), comm.next(),
          communicator::tag::halo_forward);
    }
    return owned;
  }

  static std::vector<group_type>
  halo_groups(communicator comm, std::span<T> span, stencil<1> stencl) {
    const auto &radius = stencl.radius()[0];
    std::vector<group_type> halo;
    if (stencl.periodic() || !comm.first()) {
      halo.emplace_back(span.first(radius.prev), comm.prev(),
                        communicator::tag::halo_forward);
    }
    if (stencl.periodic() || !comm.last()) {
      halo.emplace_back(span.last(radius.next), comm.next(),
                        communicator::tag::halo_reverse);
    }
    return halo;
  }
};

} // namespace lib
