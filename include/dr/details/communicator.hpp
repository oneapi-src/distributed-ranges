namespace lib {

class communicator {
public:
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

    void set_null() { win_ = MPI_INFO_NULL; }
    bool null() const noexcept { return win_ == MPI_INFO_NULL; }

    void get(void *dst, int size, int rank, int disp) const {
      drlog.debug("get::size: {}, rank: {}, win: {}, disp: {}\n", size, rank,
                  win_, disp);

      MPI_Request request;
      MPI_Rget(dst, size, MPI_CHAR, rank, disp, size, MPI_CHAR, win_, &request);
      MPI_Wait(&request, MPI_STATUS_IGNORE);
    }

    void put(const void *src, int size, int rank, int disp) const {
      drlog.debug("put:: size: {}, rank: {}, win: {}, disp: {}\n", size, rank,
                  win_, disp);
      MPI_Put(src, size, MPI_CHAR, rank, disp, size, MPI_CHAR, win_);
    }

    void fence() const {
      drlog.debug("fence:: win: {}\n", win_);
      MPI_Win_fence(0, win_);
    }

    void flush(int rank) const {
      drlog.debug("flush:: rank: {}, win: {}\n", rank, win_);
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

  MPI_Comm mpi_comm() const { return mpi_comm_; }

  void scatter(const void *src, void *dst, int size, int root) {
    MPI_Scatter(src, size, MPI_CHAR, dst, size, MPI_CHAR, root, mpi_comm_);
  }

  void gather(const void *src, void *dst, int size, int root) {
    MPI_Gather(src, size, MPI_CHAR, dst, size, MPI_CHAR, root, mpi_comm_);
  }

  void isend(const void *data, int size, int source, MPI_Request *request) {
    MPI_Isend(data, size, MPI_CHAR, source, 0, mpi_comm_, request);
  }

  template <rng::contiguous_range R>
  void isend(const R &data, int source, MPI_Request *request) {
    isend(data.data(), data.size() * sizeof(data[0]), source, request);
  }

  void irecv(void *data, int size, int dest, MPI_Request *request) {
    MPI_Irecv(data, size, MPI_CHAR, dest, 0, mpi_comm_, request);
  }

  template <rng::contiguous_range R>
  void irecv(R &data, int source, MPI_Request *request) {
    irecv(data.data(), data.size() * sizeof(data[0]), source, request);
  }

private:
  MPI_Comm mpi_comm_;
  int rank_;
  int size_;
};

} // namespace lib
