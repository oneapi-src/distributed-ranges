namespace lib {

/// Constants to specify partitions
enum class partition_method {
  /// Equal size blocks
  div
};

class block_cyclic {
public:
  /// Distribute with block_size
  ///
  block_cyclic(std::size_t block_size, MPI_Comm comm = MPI_COMM_WORLD) {
    assert(false);
  }

  /// Distribute according to partition
  ///
  block_cyclic(partition_method pm = partition_method::div,
               MPI_Comm comm = MPI_COMM_WORLD)
      : method_(pm), comm_(comm) {}

  partition_method method() { return method_; }
  MPI_Comm mpi_comm() { return comm_; }

private:
  partition_method method_;
  MPI_Comm comm_;
};

} // namespace lib
