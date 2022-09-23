template <typename T, lib::Decomposition D> class distributed_vector {
public:
  using element_type = T;

  using value_type = T;

  using size_type = std::size_t;

  /// Type of elements stored in distributed_vector
  using value_type = T;

  /// Type used for storing sizes
  using size_type = std::size_t;
  /// Type used for storing differences
  using difference_type = std::ptrdiff_t;

  /// Pointer type
  using pointer = implementation_defined;
  /// Const pointer type
  using const_pointer = implementation_defined;

  /// Reference type
  using reference = std::iter_reference_t<pointer>;
  /// Const reference type
  using const_reference = std::iter_reference_t<const_pointer>;

  /// Iterator type
  using iterator = implementation_defined;

  /// Const iterator type
  using const_iterator = implementation_defined;

  /// Construct a distributed vector with `count` elements.
  distributed_vector(size_type count, D decomp = D{});

  /// Construct a distributed vector with `count` elements equal to `value`.
  distributed_vector(size_type count, T value, D decomp);
}
