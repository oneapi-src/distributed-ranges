namespace lib {

// Halo consists of multiple segments
template <typename T> class halo {
public:
  enum class reduction_operator { SUM, REPLACE, MIN, MAX };
  class group {
    friend class halo;

  public:
    /// Construct a halo segment from a vector of indices
    group(std::size_t rank, const std::vector<std::size_t> &indices)
        : rank_(rank), indices_(indices), do_pack_(indices.size() > 1),
          op_(reduction_operator::REPLACE) {}

  private:
    std::size_t buffer_size() { return do_pack_ ? indices_.size() : 0; }

    /// Copy a group to a buffer
    void pack(T *data) {
      if (do_pack_) {
        auto b = buffer_;
        for (auto &index : indices_)
          *b++ = data[index];
      }
    }

    /// Copy a buffer to a group
    void unpack(T *data) {
      if (do_pack_) {
        auto b = buffer_;
        switch (op_) {
        case reduction_operator::REPLACE:
          for (auto &index : indices_)
            data[index] = *b++;
          break;
        case reduction_operator::SUM:
          for (auto &index : indices_)
            data[index] += *b++;
          break;
        case reduction_operator::MIN:
          for (auto &index : indices_)
            data[index] = std::min<T>(data[index], *b++);
          break;
        case reduction_operator::MAX:
          for (auto &index : indices_)
            data[index] = std::max<T>(data[index], *b++);
          break;
        }
      }
    }

    /// Asynchronous send of group
    void send(communicator comm, T *data) {
      if (do_pack_) {
        comm.isend(buffer_, indices_.size() * sizeof(T), rank_, request_);
      } else {
        comm.isend(&data[indices_[0]], sizeof(T), rank_, request_);
      }
    }

    /// Asynchronous receive of group
    void receive(communicator comm, T *data,
                 reduction_operator op = reduction_operator::REPLACE) {
      if (do_pack_) {
        op_ = op;
        comm.irecv(buffer_, indices_.size() * sizeof(T), rank_, request_);
      } else {
        assert(op == reduction_operator::REPLACE);
        comm.irecv(&data[indices_[0]], sizeof(T), rank_, request_);
      }
    }

    //// source/destination rank for this segment
    std::size_t rank_;
    std::vector<std::size_t> indices_;
    bool do_pack_;
    reduction_operator op_;
    T *buffer_ = nullptr;
    MPI_Request *request_ = nullptr;
  };

  using groups = std::vector<group>;

  void allocate_storage(auto &gs, auto &bp, auto &rp) {
    for (auto &g : gs) {
      g.request_ = rp;
      g.buffer_ = bp;

      rp++;
      bp += g.buffer_size();
    }
  }

  /// Construct a halo from groups
  halo(communicator comm, T *data, const groups &owned, const groups &halos)
      : comm_(comm), data_(data), halos_(halos), owned_(owned),
        pending_exchange_(false), pending_reduction_(false) {

    // Compute size of buffer
    std::size_t buffer_size = 0;
    for (auto &r : halos_) {
      buffer_size += r.buffer_size();
    }
    for (auto &r : owned_) {
      buffer_size += r.buffer_size();
    }

    buffer_.resize(buffer_size);
    requests_.resize(owned_.size() + halos_.size());

    auto bp = buffer_.data();
    auto rp = requests_.data();
    allocate_storage(halos_, bp, rp);
    allocate_storage(owned_, bp, rp);
  }

  /// Asynchronous halo exchange
  void exchange() {
    assert(pending_reduction_ == false);
    // Post the receives
    for (auto &receive : halos_) {
      receive.receive(comm_, data_);
    }

    // Send the data
    for (auto &send : owned_) {
      send.pack(data_);
      send.send(comm_, data_);
    }
    pending_exchange_ = true;
  }

  /// Asynchronous halo reduction
  void reduce(reduction_operator op = reduction_operator::SUM) {
    assert(pending_exchange_ == false);
    // Post the receives
    for (auto &receive : owned_) {
      receive.receive(comm_, data_, op);
    }

    // Send the data
    for (auto &send : halos_) {
      send.pack(data_);
      send.send(comm_, data_);
    }
    pending_reduction_ = true;
  }

  /// Wait for halo exchange to complete
  void wait() {
    // below assumes receives are first in requests array
    assert(owned_.size() == 0 || halos_.size() == 0 ||
           owned_[0].request_ > halos_[0].request_);
    assert(pending_exchange_ || pending_reduction_);

    int pending = requests_.size();

    while (pending > 0) {
      int completed;
      MPI_Waitany(requests_.size(), requests_.data(), &completed,
                  MPI_STATUS_IGNORE);
      if (pending_exchange_ && std::size_t(completed) < halos_.size()) {
        halos_[completed].unpack(data_);
      }
      if (pending_reduction_ && std::size_t(completed) >= halos_.size()) {
        owned_[completed - halos_.size()].unpack(data_);
      }
      pending--;
    }
    pending_exchange_ = false;
    pending_reduction_ = false;
  }

private:
  communicator comm_;
  T *data_;
  groups halos_;
  groups owned_;
  bool pending_exchange_;
  bool pending_reduction_;
  std::vector<T> buffer_;
  std::vector<MPI_Request> requests_;
};

} // namespace lib
