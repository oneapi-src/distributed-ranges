namespace lib {

// Segment of halo that targets a single rank. Segments elements are
// denoted by index and are not assumed to be contiguous
template <typename T> class halo_segment {
public:
  /// Construct a halo segment from a vector of indices
  halo_segment(std::size_t rank, const std::vector<std::size_t> &indices)
      : rank_(rank), indices_(indices) {
    finalize();
  }
  /// Construct a halo segment, indicies are provided incrementally
  halo_segment(std::size_t rank) : rank_(rank) {}

  /// Return handle to buffer
  std::vector<T> &buffer() { return buffer_; }

  /// Return a handle to vector of indices
  std::vector<std::size_t> &indices() { return indices_; }

  /// Finalize allocation after all indices are provided
  void finalize() { buffer_.resize(indices_.size()); }

  /// Copy a halo segment from data to a buffer
  template <std::random_access_iterator I> void pack(I data) {
    assert(buffer_.size() == indices_.size());
    for (std::size_t i = 0; i < buffer_.size(); i++) {
      buffer_[i] = data[indices_[i]];
    }
  }

  /// Copy a halo segment a buffer to to data
  template <std::random_access_iterator I> void unpack(I data) {
    assert(buffer_.size() == indices_.size());
    for (std::size_t i = 0; i < buffer_.size(); i++) {
      data[indices_[i]] = buffer_[i];
    }
  }

  /// Asynchronous send of halo segment
  void send(communicator comm) {
    assert(buffer_.size() == indices_.size());
    comm.isend(buffer_, rank_, &request_);
  }

  /// Asynchronous receive of halo segment
  void receive(communicator comm) {
    assert(buffer_.size() == indices_.size());
    comm.irecv(buffer_, rank_, &request_);
  }

  /// Wait for segment send/receive to complete
  void wait() { MPI_Wait(&request_, MPI_STATUS_IGNORE); }

private:
  // source/destination rank for this segment
  std::size_t rank_;
  std::vector<std::size_t> indices_;
  std::vector<T> buffer_;
  MPI_Request request_;
};

template <typename T> using halo_segments = std::vector<halo_segment<T>>;

// Halo consists of multiple segments
template <typename T> class halo : public halo_segments<T> {
public:
  /// Construct a halo from segments
  halo(communicator comm, const halo_segments<T> &segments = {})
      : halo_segments<T>(segments), comm_(comm) {}

  /// Copy halo from data to buffer
  template <std::random_access_iterator I> void pack(I data) {
    for (auto &sd : *this) {
      sd.pack(data);
    }
  }

  /// Copy halo from buffer to data
  template <std::random_access_iterator I> void unpack(I data) {
    for (auto &sd : *this) {
      sd.unpack(data);
    }
  }

  /// Asynchronous send of buffer data
  void send() {
    for (auto &sd : *this) {
      sd.send(comm_);
    }
  }

  /// Asynchronous receive of buffer data
  void receive() {
    for (auto &sd : *this) {
      sd.receive(comm_);
    }
  }

  /// Wait for send or receive to complete
  void wait() {
    for (auto &sd : *this) {
      sd.wait();
    }
  }

private:
  communicator comm_;
};

} // namespace lib
