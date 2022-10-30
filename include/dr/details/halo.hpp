namespace lib {

// Halo consists of multiple segments
template <typename T> class halo {
public:
  class group {
    friend class halo;

  public:
    /// Construct a halo segment from a vector of indices
    group(std::size_t rank, const std::vector<std::size_t> &indices)
        : rank_(rank), indices_(indices), do_pack_(indices.size() > 1) {}

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
        for (auto &index : indices_)
          data[index] = *b++;
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
    void receive(communicator comm, T *data) {
      if (do_pack_) {
        comm.irecv(buffer_, indices_.size() * sizeof(T), rank_, request_);
      } else {
        comm.irecv(&data[indices_[0]], sizeof(T), rank_, request_);
      }
    }

    //// source/destination rank for this segment
    std::size_t rank_;
    std::vector<std::size_t> indices_;
    bool do_pack_;
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
  halo(communicator comm, T *data, const groups &sends, const groups &receives)
      : comm_(comm), data_(data), receives_(receives), sends_(sends) {

    // Compute size of buffer
    std::size_t buffer_size = 0;
    for (auto &r : receives_) {
      buffer_size += r.buffer_size();
    }
    for (auto &r : sends_) {
      buffer_size += r.buffer_size();
    }

    buffer_.resize(buffer_size);
    requests_.resize(sends.size() + receives.size());

    auto bp = buffer_.data();
    auto rp = requests_.data();
    allocate_storage(receives_, bp, rp);
    allocate_storage(sends_, bp, rp);
  }

  /// Asynchronous halo exchange
  void exchange() {
    // Post the receives
    for (auto &receive : receives_) {
      receive.receive(comm_, data_);
    }

    // Send the data
    for (auto &send : sends_) {
      send.pack(data_);
      send.send(comm_, data_);
    }
  }

  /// Wait for halo exchange to complete
  void wait() {
    // below assumes receives are first in requests array
    assert(sends_.size() == 0 || receives_.size() == 0 ||
           sends_[0].request_ > receives_[0].request_);
    int pending = requests_.size();

    while (pending > 0) {
      int completed;
      MPI_Waitany(requests_.size(), requests_.data(), &completed,
                  MPI_STATUS_IGNORE);
      if (std::size_t(completed) < receives_.size()) {
        receives_[completed].unpack(data_);
      }
      pending--;
    }
  }

private:
  communicator comm_;
  T *data_;
  groups receives_;
  groups sends_;
  std::vector<T> buffer_;
  std::vector<MPI_Request> requests_;
};

} // namespace lib
