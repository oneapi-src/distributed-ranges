/// Constants to specify partitions
enum class partition { div };

/// Divide evenly
parition div();

class block_cyclic {
public:
  /// Distribute with block_size
  ///
  block_cyclic(std::size_t block_size, MPI_Comm comm = MPI_COMM_WORLD);

  /// Distribute with minimal block size sufficient to cover all the elements
  ///
  block_cyclic(partition partition, MPI_Comm comm = MPI_COMM_WORLD);
};
