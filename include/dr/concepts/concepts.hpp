#pragma once

#include "../details/ranges.hpp"
#include <ranges>

namespace lib {

template <typename I>
concept remote_iterator = std::forward_iterator<I> && requires(I &iter) {
  lib::ranges::rank(iter);
};

template <typename R>
concept remote_range = std::ranges::forward_range<R> && requires(R &r) {
  lib::ranges::rank(r);
};

template <typename R>
concept distributed_range = std::ranges::forward_range<R> && requires(R &r) {
  { lib::ranges::segments(r) } -> std::ranges::forward_range;
  { *std::ranges::begin(lib::ranges::segments(r)) } -> lib::remote_range;
};

template <typename I>
concept remote_contiguous_iterator = std::random_access_iterator<I> &&
    requires(I &iter) {
  lib::ranges::rank(iter);
  { lib::ranges::local(iter) } -> std::contiguous_iterator;
};

template <typename T>
concept remote_contiguous_range = std::ranges::random_access_range<T> &&
    /*remote_contiguous_iterator<std::ranges::iterator_t<T>> &&*/ requires(
        T t) {
  { t.rank() } -> std::convertible_to<std::size_t>;
};

template <typename T>
concept distributed_contiguous_range = std::ranges::random_access_range<T> &&
    requires(T t) {
  { t.segments() } -> std::ranges::random_access_range;
  {
    std::declval<std::ranges::range_value_t<decltype(t.segments())>>()
    } -> remote_contiguous_range;
};

} // namespace lib
