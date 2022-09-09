
template <typename T, std::size_t Extent = std::dynamic_extent>
class remote_span {
public:
  using element_type = T;
  /// Type of elements stored in span
  using value_type = std::remove_cv_t<T>;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  /// Pointer to remote data
  using pointer = remote_ptr<T>;
  /// Const pointer to remote data
  using const_pointer = remote_ptr<const T>;

  /// Proxy reference to remote data
  using reference = remote_ref<T>;
  /// Const proxy reference to remote data
  using const_reference = remote_ref<const T>;

  /// Iterator used to access data
  using iterator = pointer;

  static constexpr std::size_t extent = Extent;

  /// Construct remote_span of `count` elements beginning at `first`
  constexpr remote_span(remote_ptr<T> first, size_type count);

  /// Construct remote_span of elements in range `[first, last)`
  constexpr remote_span(remote_ptr<T> first, remote_ptr<T> last);

  constexpr remote_span() noexcept;
  constexpr remote_span(const remote_span &other) noexcept;
  constexpr remote_span &operator=(const remote_span &other);

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
  constexpr iterator begin() const noexcept;

  /// Iterator to the end of the span
  constexpr iterator end() const noexcept;

  constexpr reference front() const;

  constexpr reference back() const;

  /// Return reference to element `idx` of the span
  constexpr reference operator[](size_type idx) const;

  template <std::size_t Count>
  constexpr remote_span<element_type, Count> first() const;

  constexpr remote_span<element_type, std::dynamic_extent>
  first(size_type Count) const;

  template <std::size_t Count>
  constexpr remote_span<element_type, Count> last() const;

  constexpr remote_span<element_type, std::dynamic_extent>
  last(size_type Count) const;

  template <std::size_t Offset, std::size_t Count = std::dynamic_extent>
  constexpr auto subspan() const;

  constexpr remote_span<element_type, std::dynamic_extent>
  subspan(size_type Offset, size_type Count = std::dynamic_extent) const;
};
