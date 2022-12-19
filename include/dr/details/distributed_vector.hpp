namespace lib {

template <typename Container> struct xreference;
template <typename Container> struct const_xreference;

template <typename Container> struct const_xpointer {
  using T = typename Container::element_type;
  using const_reference = const_xreference<Container>;
  using container_type = Container;

  // Required for random access iterator
  using value_type = typename Container::value_type;
  using size_type = typename Container::size_type;
  using difference_type = typename Container::difference_type;

  // Pointer arithmetic
  bool operator==(const const_xpointer &other) const noexcept {
    return index_ == other.index_ && container_ == other.container_;
  }
  auto operator<=>(const const_xpointer &other) const noexcept {
    assert(container_ == other.container_);
    return index_ <=> other.index_;
  }

  const_xpointer &operator++() {
    index_++;
    return *this;
  }
  const_xpointer operator++(int);
  const_xpointer &operator--();
  const_xpointer operator--(int);
  difference_type operator-(const const_xpointer &other) const noexcept {
    assert(container_ == other.container_);
    return index_ - other.index_;
  }

  const_xpointer &operator-=(difference_type n);
  const_xpointer &operator+=(difference_type n);
  const_xpointer operator+(difference_type n) const noexcept {
    return const_xpointer{container_, index_ + n};
  }
  const_xpointer operator-(difference_type n) const noexcept;
  friend const_xpointer operator+(difference_type n,
                                  const const_xpointer &other) {
    return other + n;
  }

  T get() const { return container_->get(index_); }

  // dereference
  const_reference operator*() const { return const_reference{*this}; }
  const_reference operator[](difference_type n) const {
    return const_reference{*this + n};
  }

  const Container &container() { return *container_; }
  bool conforms(const const_xpointer &other) const {
    return container().conforms(other.container()) && index_ == other.index_;
  }

  const Container *container_ = nullptr;
  std::size_t index_ = 0;
};

template <typename Container> struct xpointer {
  using T = typename Container::element_type;
  using reference = xreference<Container>;
  using const_reference = const_xreference<Container>;
  using const_pointer = const_xpointer<Container>;
  using container_type = Container;

  // Required for random access iterator
  using value_type = typename Container::value_type;
  using size_type = typename Container::size_type;
  using difference_type = typename Container::difference_type;

  // Pointer arithmetic
  bool operator==(const xpointer &other) const noexcept {
    return index_ == other.index_ && container_ == other.container_;
  }
  auto operator<=>(const xpointer &other) const noexcept {
    assert(container_ == other.container_);
    return index_ <=> other.index_;
  }
  xpointer &operator++() {
    index_++;
    return *this;
  }
  xpointer operator++(int);
  xpointer &operator--();
  xpointer operator--(int);
  difference_type operator-(const xpointer &other) const noexcept {
    assert(container_ == other.container_);
    return index_ - other.index_;
  }
  xpointer &operator-=(difference_type n) const noexcept;
  xpointer &operator+=(difference_type n) const noexcept;
  xpointer operator+(difference_type n) const noexcept {
    return xpointer{container_, index_ + n};
  }
  xpointer operator-(difference_type n) const noexcept {
    return xpointer{container_, index_ - n};
  }

  friend xpointer operator+(difference_type n, const xpointer &other) {
    return other + n;
  }

  // dereference
  reference operator*() const { return reference{*this}; }
  reference operator[](difference_type n) const { return reference{*this + n}; }

  // get/set
  T get() const { return container_->get(index_); }
  void put(T val) const { container_->put(index_, val); }

  operator const_pointer() const { return const_pointer{container_, index_}; }

  Container &container() { return *container_; }
  bool conforms(xpointer other) {
    return container().conforms(other.container()) && index_ == other.index_;
  }

  auto remote_offset(int my_rank) {
    auto &local_container = container_->local();
    auto radius = container_->stencil_.radius()[0];
    auto [rank, offset] = container_->rank_offset(index_);

    // If the iterator is pointing to an earlier rank, point to the
    // beginning of my range
    if (rank < my_rank) {
      offset = radius.prev;
      // If the iterator is pointing to a later rank, point to the
      // end of my range
    } else if (rank > my_rank) {
      offset = local_container.size() - radius.next;
    }

    return offset;
  }

  auto local() {
    assert(!container_->pending_rma());

    return container_->local().begin() +
           remote_offset(container_->comm().rank());
  }

  Container *container_ = nullptr;
  std::size_t index_ = 0;
};

template <typename Container> struct const_xreference {
  using T = typename Container::element_type;
  using const_pointer = const_xpointer<Container>;

  operator T() const { return pointer_.get(); }
  const_pointer operator&() const { return pointer_; }

  const_pointer pointer_;
};

template <typename Container> struct xreference {
  using T = typename Container::element_type;
  using const_reference = const_xreference<Container>;
  using pointer = xpointer<Container>;
  using const_pointer = const_xpointer<Container>;

  operator T() const { return pointer_.get(); }
  xreference operator=(xreference &r) {
    *this = T(r);
    return *this;
  }
  xreference operator=(const T &value) const {
    pointer_.put(value);
    return *this;
  }
  pointer operator&() const { return pointer_; }
  operator const_reference() {
    return const_reference{const_pointer(pointer_)};
  }

  pointer pointer_;
};

template <typename T, typename Alloc = std::allocator<T>>
class distributed_vector {
  friend xpointer<distributed_vector>;
  friend const_xpointer<distributed_vector>;

public:
  using element_type = T;

  /// Type of elements stored in distributed_vector
  using value_type = T;

  /// Type used for storing sizes
  using size_type = std::size_t;
  /// Type used for storing differences
  using difference_type = std::ptrdiff_t;

  /// Type of local container
  using local_type = std::span<T>;

#ifdef DR_SPEC

  /// Pointer
  using pointer = implementation_defined;
  /// Pointer to immutable
  using const_pointer = implementation_defined;

#else

  // Could be GPU memory remote cpu memory. Pointer memory references
  // implemented by a constructor object or template parameter and
  // specialization

  /// Pointer type
  using pointer = xpointer<distributed_vector>;
  /// Const pointer type
  using const_pointer = const_xpointer<distributed_vector>;

#endif

  /// Reference to element
  using reference = std::iter_reference_t<pointer>;
  // using reference = std::iter_reference_t<xpointer<distributed_vector>>;
  /// Reference to immutable element
  using const_reference = std::iter_reference_t<const_pointer>;

  /// Iterator type
  using iterator = pointer;

  /// Read-only iterator
  using const_iterator = const_pointer;

  /// Stencil specification
  using stencil_type = stencil<1>;

  /// Allocator
  using allocator_type = Alloc;

  distributed_vector(const distributed_vector &) = delete;
  distributed_vector operator=(const distributed_vector &) = delete;

  /// Construct a distributed vector with a halo and `count` elements.
  distributed_vector(stencil_type s, size_type count)
      : stencil_(s), size_(count), local_(local_storage_size()),
        halo_(comm_, local_, stencil_) {
    init();
  }

  /// Construct a distributed vector with a halo and `count` elements.
  distributed_vector(stencil_type s, Alloc alloc, size_type count)
      : allocator_(alloc), stencil_(s), size_(count),
        local_(local_storage_size(), alloc), halo_(comm_, local_, stencil_) {
    init();
  }

  /// Construct a distributed vector with a halo and `count` elements.
  distributed_vector(std::size_t radius, bool periodic, size_type count)
      : stencil_(radius), size_(count), local_(local_storage_size()),
        halo_(comm_, local_, stencil_) {
    init();
  }

  /// Construct a distributed vector with `count` elements.
  distributed_vector(size_type count)
      : size_(count), local_(local_storage_size()) {
    init();
  }

  /// Construct a distributed vector with `count` elements.
  distributed_vector(Alloc alloc, size_type count)
      : allocator_(alloc), size_(count), comm_(),
        local_(local_storage_size(), alloc) {
    init();
  }

  /// Construct a distributed vector with `count` elements equal to `value`.
  distributed_vector(size_type count, T value) { assert(false); }

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

  T get(std::size_t index) const {
    auto [rank, offset] = rank_offset(index);
    assert(index < size());
    assert(offset < local().size());
    auto val = win_.get<T>(rank, offset);
    drlog.debug("get {} =  {} ({}:{})\n", val, index, rank, offset);
    return val;
  }

  void put(std::size_t index, const T &val) {
    pending_rma_ = true;
    auto [rank, offset] = rank_offset(index);
    drlog.debug("put {} ({}:{}) = {}\n", index, rank, offset, val);
    assert(index < size());
    assert(offset < local().size());
    win_.put(val, rank, offset);
  }

  /// Index into a distributed vector
  reference operator[](const size_t index) { return *iterator{this, index}; }

  /// Index into a distributed vector
  const_reference operator[](const size_t index) const {
    return *const_iterator{this, index};
  }

  iterator begin() { return iterator{this, 0}; }
  const_iterator begin() const { return const_iterator{this, 0}; }
  iterator end() { return iterator{this, size_}; };
  const_iterator end() const { return const_iterator{this, size_}; };

  void fence() {
    pending_rma_ = false;
    win_.fence();
  }

  bool pending_rma() const { return pending_rma_; }

  void flush(int rank) { win_.flush(rank); }

  const std::span<T> local() { return local_; }
  const std::span<const T> local() const { return local_; }

  const std::span<lib::remote_vector<T>> &segments() const;

  size_type size() const { return size_; }

  span_halo<T> &halo() { return halo_; }

  communicator &comm() { return comm_; }

  bool conforms(const distributed_vector &other) const noexcept {
    return size_ == other.size_;
  }

  const Alloc &allocator() const { return allocator_; }

private:
  auto rank_offset(std::size_t index) const {
    auto radius = stencil_.radius()[0];
    std::size_t rank, offset;
    if (index < radius.prev) {
      rank = 0;
    } else if (index >= size() - radius.next) {
      rank = comm_.size() - 1;
    } else {
      rank = (index - radius.prev) / slice_size_;
    }
    offset = index - rank * slice_size_;

    return std::tuple<int, std::size_t>(rank, offset);
  }

  void init() {
    auto radius = stencil_.radius()[0];
    slice_size_ = local().size() - radius.prev - radius.next;

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

  Alloc allocator_;
  stencil_type stencil_;
  size_type size_;
  size_type slice_size_;
  communicator comm_;
  std::vector<T, Alloc> local_;
  communicator::win win_;
  span_halo<T> halo_;
  bool pending_rma_ = false;
};

} // namespace lib
