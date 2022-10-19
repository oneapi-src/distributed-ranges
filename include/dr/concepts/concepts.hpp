namespace lib {

template <typename I>
concept remote_contiguous_iterator = std::random_access_iterator<I> &&
    requires(I i) {
  { i.rank() } -> std::convertible_to<std::size_t>;
  { i.local() } -> std::contiguous_iterator;
};

template <typename T>
concept remote_contiguous_range = std::ranges::random_access_range<T> &&
    /*remote_contiguous_iterator<std::ranges::iterator_t<T>> &&*/ requires(
        T t) {
  { t.rank() } -> std::convertible_to<std::size_t>;
};

template <typename T>
concept distributed_contiguous_range = rng::random_access_range<T> &&
    requires(T t) {
  { t.segments() } -> std::ranges::random_access_range;
  {
    std::declval<std::ranges::range_value_t<decltype(t.segments())>>()
    } -> remote_contiguous_range;
};

} // namespace lib
