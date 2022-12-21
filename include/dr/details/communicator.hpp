// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

namespace lib {

class communicator {
public:
  enum class tag {
    invalid,
    halo_forward,
    halo_reverse,
    halo_index,
  };

  class win {
  public:
    win() { win_ = MPI_WIN_NULL; }

    void create(const communicator &comm, void *data, int size) {
      MPI_Win_create(data, size, 1, MPI_INFO_NULL, comm.mpi_comm(), &win_);
    }

    void free() { MPI_Win_free(&win_); }

    bool operator==(const win other) const noexcept {
      return this->win_ == other.win_;
    }

    void set_null() { win_ = MPI_WIN_NULL; }
    bool null() const noexcept { return win_ == MPI_WIN_NULL; }

    template <typename T> T get(int rank, int disp) const {
      T dst;
      get(&dst, sizeof(T), rank, disp * sizeof(T));
      return dst;
    }

    void get(void *dst, int size, int rank, int disp) const {
      MPI_Request request;
      MPI_Rget(dst, size, MPI_CHAR, rank, disp, size, MPI_CHAR, win_, &request);
      MPI_Wait(&request, MPI_STATUS_IGNORE);
    }

    void put(const auto &src, int rank, int disp) const {
      put(&src, sizeof(src), rank, disp * sizeof(src));
    }

    void put(const void *src, int size, int rank, int disp) const {
      MPI_Put(src, size, MPI_CHAR, rank, disp, size, MPI_CHAR, win_);
    }

    void fence() const {
      drlog.debug("fence\n");
      MPI_Win_fence(0, win_);
    }

    void flush(int rank) const {
      drlog.debug("flush:: rank: {}\n", rank);
      MPI_Win_flush(rank, win_);
    }

  private:
    MPI_Win win_;
  };

  communicator(MPI_Comm comm = MPI_COMM_WORLD) {
    mpi_comm_ = comm;
    MPI_Comm_rank(comm, &rank_);
    MPI_Comm_size(comm, &size_);
  }

  int size() const { return size_; }
  int rank() const { return rank_; }
  int prev() const { return (rank() + size() - 1) % size(); }
  int next() const { return (rank() + 1) % size(); }
  int first() const { return rank() == 0; }
  int last() const { return rank() == size() - 1; }

  MPI_Comm mpi_comm() const { return mpi_comm_; }

  void scatter(const void *src, void *dst, int size, int root) {
    MPI_Scatter(src, size, MPI_CHAR, dst, size, MPI_CHAR, root, mpi_comm_);
  }

  void scatterv(const void *src, int *counts, int *offsets, void *dst,
                int root) {
    MPI_Scatterv(src, counts, offsets, MPI_CHAR, dst, counts[rank()], MPI_CHAR,
                 root, mpi_comm_);
  }

  void gather(const void *src, void *dst, int size, int root) {
    MPI_Gather(src, size, MPI_CHAR, dst, size, MPI_CHAR, root, mpi_comm_);
  }

  template <typename T>
  void gather(const T &src, std::vector<T> &dst, int root) {
    dst.resize(size());
    MPI_Gather(&src, sizeof(src), MPI_CHAR, dst.data(), sizeof(src), MPI_CHAR,
               root, mpi_comm_);
  }

  void gatherv(const void *src, int *counts, int *offsets, void *dst,
               int root) {
    MPI_Gatherv(src, counts[rank()], MPI_CHAR, dst, counts, offsets, MPI_CHAR,
                root, mpi_comm_);
  }

  template <typename T>
  void isend(const T *data, int size, int source, tag t, MPI_Request *request) {
    MPI_Isend(data, size * sizeof(T), MPI_CHAR, source, int(t), mpi_comm_,
              request);
  }

  template <rng::contiguous_range R>
  void isend(const R &data, int source, tag t, MPI_Request *request) {
    isend(data.data(), data.size(), source, int(t), request);
  }

  template <typename T>
  void irecv(T *data, int size, int dest, tag t, MPI_Request *request) {
    MPI_Irecv(data, size * sizeof(T), MPI_CHAR, dest, int(t), mpi_comm_,
              request);
  }

  template <rng::contiguous_range R>
  void irecv(R &data, int source, tag t, MPI_Request *request) {
    irecv(data.data(), data.size(), source, int(t), request);
  }

  bool operator==(const communicator &other) const {
    return mpi_comm_ == other.mpi_comm_;
  }

private:
  MPI_Comm mpi_comm_;
  int rank_;
  int size_;
};

} // namespace lib
