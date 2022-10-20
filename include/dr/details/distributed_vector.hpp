namespace lib {

template <typename T, typename D = block_cyclic> class distributed_vector {
  using rptr = remote_pointer<T>;

public:
  using element_type = T;

  /// Type of elements stored in distributed_vector
  using value_type = T;

  /// Type used for storing sizes
  using size_type = std::size_t;
  /// Type used for storing differences
  using difference_type = std::ptrdiff_t;

  /// Type of decomposition
  using decomposition = D;

#ifdef DR_SPEC

  /// Pointer type
  using pointer = implementation_defined;
  /// Const pointer type
  using const_pointer = implementation_defined;

#else

  // Could be GPU memory remote cpu memory. Pointer memory references
  // implemented by a constructor object or template parameter and
  // specialization

  /// Pointer type
  using pointer = rptr;
  /// Const pointer type
  using const_pointer = const pointer;

#endif

  /// Reference type
  using reference = std::iter_reference_t<pointer>;
  /// Const reference type
  using const_reference = std::iter_reference_t<const_pointer>;

  // Placeholder

#ifdef DR_SPEC

  /// Iterator type
  using iterator = implementation_defined;

  /// Const iterator type
  using const_iterator = implementation_defined;

#else

  // placeholder
  /// Iterator type
  using iterator = T *;

  /// Const iterator type
  using const_iterator = const iterator;

#endif

  /// Construct a distributed vector with `count` elements.
  distributed_vector(size_type count, D decomp = D{})
      : comm_(decomp.mpi_comm()) {
    assert(decomp.method() == partition_method::div);
    auto sz = partition_up(count, comm_.size());
    local_segment_ = std::span(new T[sz], sz);
    // win_ = comm_.win_create(local_data_, local_size_ * sizeof(T));
  }

  /// Construct a distributed vector with `count` elements equal to `value`.
  distributed_vector(size_type count, T value, D decomp) { assert(false); }

  ~distributed_vector() { delete[] local_segment_.data(); }

  /// copy a span to a distributed vector
  void scatter(const std::span<T> src, int root) {
    assert(comm_.rank() != root ||
           comm_.size() * local_segment_.size() == src.size());
    comm_.scatter(src.data(), local_segment_.data(),
                  local_segment_.size() * sizeof(T), root);
  }

  /// copy a distributed vector to a span
  void gather(const std::span<T> dst, int root) {
    assert(comm_.rank() != root ||
           comm_.size() * local_segment_.size() == dst.size());
    comm_.gather(local_segment_.data(), dst.data(),
                 local_segment_.size() * sizeof(T), root);
  }

  /// Index into a distributed vector
  T &operator[](const size_t index);

  iterator begin() const;
  iterator end() const;

  const std::span<T> local_segment() const { return local_segment_; }

  const std::span<lib::remote_vector<T>> &segments() const;

private:
  MPI_Win win_;
  communicator comm_;
  std::span<T> local_segment_;
};

} // namespace lib
