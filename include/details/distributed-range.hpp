///
/// Distributed range
///
/// @tparam T element type
///
template <typename T> struct distributed_range {

  distributed_range() {}

  ///
  /// Shape of range
  ///
  size_t shape() { return 1; }
};
