template <typename T> class remote_reference;

template <typename T> class remote_pointer {
  // static_assert(std::forward_iterator<remote_pointer<int>>);
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

  remote_pointer(std::size_t rank, std::size_t window, std::size_t offset)
      : rank_(rank), window_(window), offset_(offset) {}

  remote_pointer(std::nullptr_t null) : rank_(0), window_(0), offset_(0) {}

  // TODO: convert SFINAE to requires() for C++20
  // template <__BCL_REQUIRES(!std::is_same_v<std::decay_t<T>, void> &&
  //! std::is_const_v<T> &&
  //! std::is_same_v<T, void>)>
  operator remote_pointer<void>() const noexcept {
    return remote_pointer<void>(rank_, window_, offset_);
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
    return const_pointer(rank_, window_, offset_);
  }

  remote_pointer &operator=(std::nullptr_t null) {
    rank_ = 0;
    window_ = 0;
    offset_ = 0;
    return *this;
  }

  bool operator==(const remote_pointer<T> other) const noexcept {
    return (rank_ == other.rank_ && offset_ == other.offset_ &&
            window_ == other.window_);
  }

  bool operator!=(const_pointer other) const noexcept {
    return !(*this == other);
  }

  bool operator==(std::nullptr_t null) const noexcept {
    return (rank_ == 0 && offset_ == 0 && window_ == 0);
  }

  bool operator!=(std::nullptr_t null) const noexcept {
    return !(*this == null);
  }

  /// Dereference the global pointer, returning a global reference `GlobalRef`
  /// that can be used to read or write to the memory location.
  reference operator*() const noexcept { return reference(*this); }

  reference operator[](difference_type offset) const noexcept {
    return *(*this + offset);
  }

  pointer operator+(difference_type offset) const noexcept {
    // pointer operator+(long long int offset) const noexcept {
    return pointer(rank_, window_, offset_ + offset * sizeof(T));
  }

  pointer operator-(difference_type offset) const noexcept {
    return pointer(rank_, window_, offset_ - offset * sizeof(T));
  }

  difference_type operator-(const_pointer other) const noexcept {
    return (offset_ - difference_type(other.offset_)) / sizeof(T);
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
    offset_ += sizeof(T);
    return *this;
  }

  pointer operator++(int) noexcept {
    pointer other(*this);
    ++(*this);
    return other;
  }

  pointer &operator--() noexcept {
    offset_ -= sizeof(T);
    return *this;
  }

  pointer operator--(int) noexcept {
    pointer other(*this);
    --(*this);
    return other;
  }

  pointer &operator+=(difference_type offset) noexcept {
    offset_ += offset * sizeof(T);
    return *this;
  }

  pointer &operator-=(difference_type offset) noexcept {
    offset_ -= offset * sizeof(T);
    return *this;
  }

  friend pointer operator+(difference_type n, pointer iter) { return iter + n; }

  T *local() { return local_pointer_; }

  std::size_t rank() { return rank_; }

private:
  std::size_t rank_;
  std::size_t window_;
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
  remote_reference &operator=(remote_reference &&) = default;

  using value_type = T;
  using pointer = remote_pointer<T>;
  using reference = remote_reference<T>;
  using const_reference = remote_reference<std::add_const_t<T>>;

  remote_reference(remote_pointer<T> ptr) : pointer_(ptr) {
    assert(ptr != nullptr);
  }

  operator T() const;

  operator const_reference() const;

  reference operator=(const T &value) const;

  pointer operator&() const noexcept { return pointer_; }

private:
  remote_pointer<T> pointer_ = nullptr;
};
