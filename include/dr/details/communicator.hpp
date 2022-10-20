namespace lib {

class communicator {
public:
  communicator(MPI_Comm comm = MPI_COMM_WORLD) {
    mpi_comm_ = comm;
    MPI_Comm_rank(comm, &rank_);
    MPI_Comm_size(comm, &size_);
  }

  int size() { return size_; }
  int rank() { return rank_; }

  MPI_Win win_create(void *data, size_t size) {
    MPI_Win win;
    MPI_Win_create(data, size, 1, MPI_INFO_NULL, mpi_comm_, &win);
    return win;
  }

  MPI_Comm mpi_comm() { return mpi_comm_; }

  void scatter(const void *src, void *dst, int size, int root) {
    MPI_Scatter(src, size, MPI_CHAR, dst, size, MPI_CHAR, root, mpi_comm_);
  }

  void gather(const void *src, void *dst, int size, int root) {
    MPI_Gather(src, size, MPI_CHAR, dst, size, MPI_CHAR, root, mpi_comm_);
  }

private:
  MPI_Comm mpi_comm_;
  int rank_;
  int size_;
};

} // namespace lib
