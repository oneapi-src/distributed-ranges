namespace lib {

template <typename T> class remote_vector {
public:
  using rptr = remote_pointer<T>;
  // using rptr = T *;

  using element_type = T;
  /// Type of elements stored in span
  using value_type = std::remove_cv_t<T>;

  using size_type = std::size_t;
  // placeholder
  using difference_type = std::ptrdiff_t;

  // placeholder
  /// Pointer to remote data
  using pointer = rptr;

  /// Proxy reference to remote data
  using reference = std::iter_reference_t<rptr>;

  // placeholder
  /// Iterator used to access data
  using iterator = rptr;

  remote_vector() {
    base_ = 0;
    size_ = 0;
  }

  ~remote_vector() {}

  /// Number of elements in span
  constexpr size_type size();

  /// Size of span in bytes
  constexpr size_type size_bytes() const noexcept;

  /// Whether the span is empty
  [[nodiscard]] constexpr bool empty() const noexcept;

  /// The rank of the process on which this span is located.
  size_type rank() const noexcept;

  /// Pointer to the beginning of the span
  constexpr pointer data() const noexcept;

  /// Iterator to the beginning of the span
  constexpr iterator begin() const noexcept { return base_; }

  /// Iterator to the end of the span
  constexpr iterator end() const noexcept { return base_ + size_; }

  /// First element in span
  constexpr reference front() const;

  /// Last element in span
  constexpr reference back() const;

  /// Return reference to element `idx` of the span
  constexpr reference operator[](size_type idx) const;

private:
  pointer base_;
  size_type size_;
};

} // namespace lib
