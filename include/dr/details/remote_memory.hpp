namespace lib {

template <typename T> class remote_reference;

template <typename T> class remote_pointer {
  friend class remote_reference<T>;

public:
  using difference_type = std::ptrdiff_t;
  using value_type = T;
  using pointer = remote_pointer<T>;
  using const_pointer = remote_pointer<std::add_const_t<T>>;
  using reference = remote_reference<T>;
  using iterator_category = std::random_access_iterator_tag;

  remote_pointer() = default;
  ~remote_pointer() = default;
  remote_pointer(const remote_pointer &) = default;
  remote_pointer &operator=(const remote_pointer &) = default;
  remote_pointer(remote_pointer &&) = default;
  remote_pointer &operator=(remote_pointer &&) = default;

  remote_pointer(std::size_t rank, communicator::win win, std::size_t offset)
      : rank_(rank), win_(win), offset_(offset) {}

  remote_pointer(std::nullptr_t null) : rank_(0), win_(), offset_(0) {}

  // TODO: convert SFINAE to requires() for C++20
  // template <__BCL_REQUIRES(!std::is_same_v<std::decay_t<T>, void> &&
  //! std::is_const_v<T> &&
  //! std::is_same_v<T, void>)>
  operator remote_pointer<void>() const noexcept {
    return remote_pointer<void>(rank_, win_, offset_);
  }

  // TODO: convert SFINAE to requires() for C++20
  //       Extra checks for Intel compiler
  // template <__BCL_REQUIRES(!std::is_same_v<std::decay_t<T>, void> &&
  //! std::is_same_v<T, void> &&
  //! std::is_same_v<T, const void>)>
  // operator remote_pointer<const void>() const noexcept {
  // return remote_pointer<const void>(rank, ptr);
  // }

  // template <__BCL_REQUIRES(!std::is_const_v<T>)>
  operator const_pointer() const noexcept {
    return const_pointer(rank_, win_, offset_);
  }

  remote_pointer &operator=(std::nullptr_t null) {
    rank_ = 0;
    win_.set_null();
    offset_ = 0;
    return *this;
  }

  bool operator==(const remote_pointer<T> other) const noexcept {
    return (rank_ == other.rank_ && offset_ == other.offset_ &&
            win_ == other.win_);
  }

  bool operator!=(const_pointer other) const noexcept {
    return !(*this == other);
  }

  bool operator==(std::nullptr_t null) const noexcept {
    return (rank_ == 0 && offset_ == 0 && win_.null());
  }

  bool operator!=(std::nullptr_t null) const noexcept {
    return !(*this == null);
  }

  /// Dereference the global pointer, returning a global reference `GlobalRef`
  /// that can be used to read or write to the memory location.
  reference operator*() const noexcept {
    drlog.debug(nostd::source_location::current(), "remote pointer *\n");
    return reference(*this);
  }

  reference operator[](difference_type offset) const noexcept {
    return *(*this + offset);
  }

  pointer operator+(difference_type offset) const noexcept {
    return pointer(rank_, win_, offset_ + offset);
  }

  pointer operator-(difference_type offset) const noexcept {
    return pointer(rank_, win_, offset_ - offset);
  }

  difference_type operator-(const_pointer other) const noexcept {
    return (offset_ - difference_type(other.offset_));
  }

  bool operator<(const_pointer other) const noexcept {
    return offset_ < other.offset_;
  }

  bool operator>(const_pointer other) const noexcept {
    return offset_ > other.offset_;
  }

  bool operator<=(const_pointer other) const noexcept {
    return offset_ <= other.offset_;
  }

  bool operator>=(const_pointer other) const noexcept {
    return offset_ >= other.offset_;
  }

  pointer &operator++() noexcept {
    offset_++;
    return *this;
  }

  pointer operator++(int) noexcept {
    pointer other(*this);
    ++(*this);
    return other;
  }

  pointer &operator--() noexcept {
    offset_--;
    return *this;
  }

  pointer operator--(int) noexcept {
    pointer other(*this);
    --(*this);
    return other;
  }

  pointer &operator+=(difference_type offset) noexcept {
    offset_ += offset;
    return *this;
  }

  pointer &operator-=(difference_type offset) noexcept {
    offset_ -= offset;
    return *this;
  }

  friend pointer operator+(difference_type n, pointer iter) { return iter + n; }

  T *local() { return local_pointer_; }

  std::size_t rank() { return rank_; }

private:
  std::size_t rank_;
  communicator::win win_;
  std::size_t offset_;
  T *local_pointer_;
};

template <typename T> class remote_reference {
public:
  remote_reference() = delete;
  ~remote_reference() = default;
  remote_reference(const remote_reference &) = default;
  remote_reference &operator=(const remote_reference &) = default;
  remote_reference(remote_reference &&) = default;
  remote_reference &operator=(remote_reference &r) {
    drlog.debug(nostd::source_location::current(), "remote ref operator=\n");
    *this = T(r);
    return *this;
  }

  using value_type = T;
  using pointer = remote_pointer<T>;
  using reference = remote_reference<T>;
  using const_reference = remote_reference<std::add_const_t<T>>;

  remote_reference(remote_pointer<T> ptr) : pointer_(ptr) {
    assert(ptr != nullptr);
  }

  operator T() const {
    T value;
    pointer_.win_.get(&value, sizeof(T), pointer_.rank_,
                      pointer_.offset_ * sizeof(T));
    drlog.debug(nostd::source_location::current(), "get: {}\n", value);
    return value;
  }

  operator const_reference() const;

  reference operator=(const T &value) const {
    drlog.debug(nostd::source_location::current(), "put: {}\n", value);
    pointer_.win_.put(&value, sizeof(T), pointer_.rank_,
                      pointer_.offset_ * sizeof(T));
    return pointer_;
  }

  pointer operator&() const noexcept { return pointer_; }

private:
  remote_pointer<T> pointer_ = nullptr;
};

} // namespace lib
