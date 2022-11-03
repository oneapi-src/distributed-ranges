namespace lib {

// Make a random access iterator for an object that supports index
// reference
template <typename O> class index_iterator {
public:
  using value_type = typename O::value_type;
  using size_type = typename O::size_type;
  using difference_type = typename O::difference_type;
  using reference = typename O::reference;

  index_iterator();
  index_iterator(O &o, size_type index) : o_(&o), index_(index) {}

  reference operator*() const { return (*o_)[index_]; }
  reference operator*() { return (*o_)[index_]; }
  reference operator[](difference_type n) const noexcept {
    return (*o_)[index_ + n];
  }
  reference operator[](difference_type n) { return (*o_)[index_ + n]; }

  bool operator==(const index_iterator &other) const noexcept {
    return index_ == other.index_ && o_ == other.o_;
  }

  bool operator<=>(const index_iterator &other) const noexcept;

  index_iterator &operator++() {
    index_++;
    return *this;
  }
  index_iterator operator++(int);
  index_iterator &operator--();
  index_iterator operator--(int);
  difference_type operator-(const index_iterator &other) const noexcept;
  index_iterator &operator-=(difference_type n) const noexcept;
  index_iterator &operator+=(difference_type n) const noexcept;
  index_iterator operator+(difference_type n) const noexcept;
  index_iterator operator-(difference_type n) const noexcept;

  friend index_iterator operator+(difference_type n,
                                  const index_iterator &other) {
    return other + n;
  }

private:
  O *o_ = nullptr;
  difference_type index_ = 0;
};

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
  using iterator = index_iterator<distributed_vector>;

  /// Const iterator type
  using const_iterator = const iterator;

#endif

  /// Construct a distributed vector with `count` elements.
  distributed_vector(size_type count, D decomp = D{})
      : size_(count), comm_(decomp.mpi_comm()),
        local_segment_(partition_up(count, comm_.size())) {
    win_.create(comm_, local_segment_.data(),
                local_segment_.size() * sizeof(T));
    assert(decomp.method() == partition_method::div);
  }

  /// Construct a distributed vector with `count` elements equal to `value`.
  distributed_vector(size_type count, T value, D decomp = D{}) {
    assert(false);
  }

  ~distributed_vector() { win_.free(); }

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
  reference operator[](const size_t index) {
    return rptr(index / local_segment_.size(), win_,
                index % local_segment_.size());
  }

  iterator begin() { return iterator(*this, 0); }
  const iterator begin() const { return iterator(*this, 0); }
  iterator end() { return iterator(*this, size_); };
  const iterator end() const { return iterator(*this, size_); };

  void fence() { win_.fence(); }

  void flush(int rank) { win_.flush(rank); }

  const std::span<T> local_segment() { return std::span(local_segment_); }

  const std::span<lib::remote_vector<T>> &segments() const;

  size_type size() const { return size_; }

private:
  size_type size_;
  communicator comm_;
  std::vector<T> local_segment_;
  communicator::win win_;
};

} // namespace lib
