/// Constants to specify partitions
enum class partition {
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
  block_cyclic(partition partition, MPI_Comm comm = MPI_COMM_WORLD)
      : partition_(partition), comm_(comm) {}

  partition part() { return partition_; }
  MPI_Comm comm() { return comm_; }

private:
  partition partition_;
  MPI_Comm comm_;
};
