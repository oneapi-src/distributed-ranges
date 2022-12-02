namespace lib {

template <typename O> class const_index_iterator;

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

  // Restrict to friends?
  O &object() { return *o_; }

private:
  friend const_index_iterator<O>;
  O *o_ = nullptr;
  difference_type index_ = 0;
};

template <typename O> class const_index_iterator {
public:
  using value_type = typename O::value_type;
  using size_type = typename O::size_type;
  using difference_type = typename O::difference_type;
  using const_reference = typename O::const_reference;

  const_index_iterator();
  const_index_iterator(const O &o, size_type index) : o_(&o), index_(index) {}

  const_reference operator*() const { return (*o_)[index_]; }

  const_reference operator[](difference_type n) const noexcept {
    return (*o_)[index_ + n];
  }

  bool operator==(const index_iterator<O> &other) const noexcept {
    return index_ == other.index_ && o_ == other.o_;
  }
  bool operator==(const const_index_iterator<O> &other) const noexcept {
    return index_ == other.index_ && o_ == other.o_;
  }

  bool operator<=>(const index_iterator<O> &other) const noexcept;
  bool operator<=>(const const_index_iterator<O> &other) const noexcept;

  const_index_iterator &operator++() {
    index_++;
    return *this;
  }
  const_index_iterator operator++(int);
  const_index_iterator &operator--();
  const_index_iterator operator--(int);
  difference_type operator-(const index_iterator<O> &other) const noexcept;
  difference_type
  operator-(const const_index_iterator<O> &other) const noexcept;
  const_index_iterator &operator-=(difference_type n) const noexcept;
  const_index_iterator &operator+=(difference_type n) const noexcept;
  const_index_iterator operator+(difference_type n) const noexcept;
  const_index_iterator operator-(difference_type n) const noexcept;

  friend const_index_iterator operator+(difference_type n,
                                        const_index_iterator &other) {
    return other + n;
  }

private:
  friend index_iterator<O>;

  const O *o_ = nullptr;
  difference_type index_ = 0;
};

template <typename T, typename Alloc = std::allocator<T>,
          typename D = block_cyclic>
class distributed_vector {
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
  using const_iterator = const const_index_iterator<distributed_vector>;

#endif

  /// Stencil specification
  using stencil_type = stencil<1>;

  /// Allocator
  using allocator_type = Alloc;

  /// Construct a distributed vector with `count` elements.
  distributed_vector(D decomp, size_type count)
      : decomp_(decomp), size_(count), comm_(decomp.comm()),
        local_(local_storage_size()) {
    init();
  }

  /// Construct a distributed vector with a halo and `count` elements.
  distributed_vector(stencil_type s, size_type count)
      : stencil_(s), size_(count), comm_(decomp_.comm()),
        local_(local_storage_size()), halo_(comm_, local_, stencil_) {
    init();
  }

  /// Construct a distributed vector with a halo and `count` elements.
  distributed_vector(stencil_type s, Alloc alloc, size_type count)
      : stencil_(s), size_(count), comm_(decomp_.comm()),
        local_(local_storage_size(), alloc), halo_(comm_, local_, stencil_) {
    init();
  }

  /// Construct a distributed vector with a halo and `count` elements.
  distributed_vector(std::size_t radius, bool periodic, size_type count)
      : stencil_(radius), size_(count), comm_(decomp_.comm()),
        local_(local_storage_size()), halo_(comm_, local_, stencil_) {
    init();
  }

  /// Construct a distributed vector with `count` elements.
  distributed_vector(size_type count)
      : size_(count), comm_(decomp_.comm()), local_(local_storage_size()) {
    init();
  }

  /// Construct a distributed vector with `count` elements.
  distributed_vector(Alloc alloc, size_type count)
      : size_(count), comm_(decomp_.comm()),
        local_(local_storage_size(), alloc) {
    init();
  }

  /// Construct a distributed vector with `count` elements equal to `value`.
  distributed_vector(size_type count, T value, D decomp = D{}) {
    assert(false);
  }

  ~distributed_vector() {
    fence();
    win_.free();
  }

  /// copy a span to a distributed vector
  void scatter(const std::span<T> src, int root) {
    assert(comm_.rank() != root || comm_.size() * local_.size() == src.size());
    comm_.scatter(src.data(), local_.data(), local_.size() * sizeof(T), root);
  }

  /// copy a distributed vector to a span
  void gather(const std::span<T> dst, int root) {
    assert(comm_.rank() != root || comm_.size() * local_.size() == dst.size());
    comm_.gather(local_.data(), dst.data(), local_.size() * sizeof(T), root);
  }

  /// Index into a distributed vector
  reference operator[](const size_t index) {
    auto [rank, offset] = rank_offset(index);
    return *rptr(rank, win_, offset);
  }

  /// Index into a distributed vector
  const_reference operator[](const size_t index) const {
    auto [rank, offset] = rank_offset(index);
    return *rptr(rank, win_, offset);
  }

  iterator begin() { return iterator(*this, 0); }
  const_iterator begin() const { return const_iterator(*this, 0); }
  iterator end() { return iterator(*this, size_); };
  const_iterator end() const { return const_iterator(*this, size_); };

  void fence() { win_.fence(); }

  void flush(int rank) { win_.flush(rank); }

  const std::span<T> local() { return local_; }
  const std::span<const T> local() const { return local_; }

  const std::span<lib::remote_vector<T>> &segments() const;

  size_type size() const { return size_; }

  span_halo<T> &halo() { return halo_; }

  communicator &comm() { return comm_; }

  bool conforms(const distributed_vector &other) const noexcept {
    return decomp_ == other.decomp_ && size_ == other.size_;
  }

  bool congruent(const iterator &first, const iterator &last) const noexcept {
    return first == begin() && last == end();
  }

  bool congruent(const iterator &first) const noexcept {
    return first == begin();
  }

private:
  auto rank_offset(std::size_t index) const {
    auto radius = stencil_.radius()[0];
    std::size_t slice_size = local().size() - radius.prev - radius.next;
    std::size_t rank, offset;
    if (index < radius.prev) {
      rank = 0;
    } else if (index >= size() - radius.next) {
      rank = comm_.size() - 1;
    } else {
      rank = (index - radius.prev) / slice_size;
    }
    offset = index - rank * slice_size;

    return std::pair(rank, offset);
  }

  void init() {
#ifdef OMPI_MAJOR_VERSION
    // openmpi cannot create a window when the size is 1
    // Not sure if it is specific to the version that comes with ubuntu 22
    assert(comm_.size() > 1);
#endif
    win_.create(comm_, local_.data(), local_.size() * sizeof(T));
    fence();
  }

  auto local_storage_size() const {
    auto radius = stencil_.radius()[0];
    return partition_up(size_ - radius.prev - radius.next, comm_.size()) +
           radius.prev + radius.next;
  }

  stencil_type stencil_;
  D decomp_;
  size_type size_;
  communicator comm_;
  std::vector<T, Alloc> local_;
  communicator::win win_;
  span_halo<T> halo_;
};

} // namespace lib
