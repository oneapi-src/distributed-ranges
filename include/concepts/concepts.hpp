#pragma once

#include <concepts>
#include <iterator>
#include <ranges>

/// A remote contiguous iterator acts as a pointer to some contiguous piece
/// of remote memory. It is an `std::random_access_iterator` and has a method
/// `local` that returns a local pointer of some kind.
/// Instantiations of `remote_ptr`, `device_ptr`, and `BCL::GlobalPtr` all
/// fulfill this concept.
template <typename I>
concept remote_contiguous_iterator = std::random_access_iterator<I> &&
    requires(I i) {
  { i.local() } -> std::contiguous_iterator;
  { i.rank() } -> std::convertible_to<std::size_t>;
};

/// A remote contiguous range represents a contiguous range of remote memory.
///
/// 1) It is a standard `random_access_range`
///
/// 2) Its iterator is a `remote_contiguous_iterator`
///
/// 3) There is a mechanism for determining on which rank the contiguous remote
/// range is located.
template <typename T>
concept remote_contiguous_range = std::ranges::random_access_range<T> &&
    remote_contiguous_iterator<std::ranges::iterator_t<T>> && requires(T t) {
  { t.rank() } -> std::convertible_to<std::size_t>;
};

/// A distributed contiguous range represents a distributed range of contiguous
/// memory.
///
/// 1) It is a standard `random_access_range`
///
/// 2) It has a mechanism for retrieving all of the `subranges` that make up the
/// distributed range
///
/// 3) Each of the subranges is a `remote_contiguous_range`.
template <typename T>
concept distributed_contiguous_range = std::ranges::random_access_range<T> &&
    requires(T t) {
  { t.subranges() } -> std::ranges::random_access_range {
      std::declval<std::ranges::range_value_t<decltype(t.subranges())>>()
      } -> remote_contiguous_range;
};
