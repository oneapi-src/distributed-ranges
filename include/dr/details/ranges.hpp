#pragma once

#include <iterator>
#include <ranges>
#include <type_traits>

namespace lib {

namespace ranges {

template <typename> inline constexpr bool disable_rank = false;

namespace {

template <typename T>
concept has_rank_method = requires(T t) {
                            { t.rank() } -> std::weakly_incrementable;
                          };

template <typename Iter>
concept is_remote_iterator_shadow_impl_ =
    std::forward_iterator<Iter> && has_rank_method<Iter> && !
disable_rank<std::remove_cv_t<Iter>>;

} // namespace

namespace {

struct rank_fn_ {

  // Return the rank associated with a remote range.
  // This is either:
  // 1) r.rank(), if the remote range has a `rank()` method
  // OR, if not available,
  // 2) r.begin().rank(), if iterator is `remote_iterator`
  template <std::ranges::forward_range R>
    requires((has_rank_method<R> && !disable_rank<std::remove_cv_t<R>>) ||
             is_remote_iterator_shadow_impl_<std::ranges::range_value_t<R>>)
  constexpr auto operator()(R &&r) const {
    if constexpr (has_rank_method<R> && !disable_rank<std::remove_cv_t<R>>) {
      return std::forward<R>(r).rank();
    } else if constexpr (is_remote_iterator_shadow_impl_<
                             std::ranges::range_value_t<R>>) {
      return operator()(std::ranges::begin(std::forward<R>(r)));
    }
  }

  template <std::forward_iterator Iter>
    requires(has_rank_method<Iter> && !disable_rank<std::remove_cv_t<Iter>>)
  auto operator()(Iter iter) const {
    if constexpr (has_rank_method<Iter> &&
                  !disable_rank<std::remove_cv_t<Iter>>) {
      return std::forward<Iter>(iter).rank();
    }
  }
};

} // namespace

const inline auto rank = rank_fn_{};

namespace {

template <typename R>
concept remote_range_shadow_impl_ =
    std::ranges::forward_range<R> && requires(R &r) { lib::ranges::rank(r); };

template <typename R>
concept segments_range =
    std::ranges::forward_range<R> &&
    remote_range_shadow_impl_<std::ranges::range_value_t<R>>;

template <typename R>
concept has_segments_method = requires(R r) {
                                { r.segments() } -> segments_range;
                              };

struct segments_fn_ {
  template <std::ranges::forward_range R>
    requires(has_segments_method<R>)
  constexpr decltype(auto) operator()(R &&r) const {
    return std::forward<R>(r).segments();
  }
};

} // namespace

const inline auto segments = segments_fn_{};

namespace {

template <typename Iter>
concept has_local_method =
    std::forward_iterator<Iter> && requires(Iter i) {
                                     { i.local() } -> std::forward_iterator;
                                   };

struct local_fn_ {

  template <std::forward_iterator Iter>
    requires(has_local_method<Iter> || std::contiguous_iterator<Iter>)
  auto operator()(Iter iter) const {
    if constexpr (has_local_method<Iter>) {
      return iter.local();
    } else if constexpr (std::contiguous_iterator<Iter>) {
      return iter;
    }
  }

  template <std::ranges::forward_range R>
    requires(has_local_method<std::ranges::iterator_t<R>> ||
             std::contiguous_iterator<std::ranges::iterator_t<R>> ||
             std::ranges::contiguous_range<R>)
  auto operator()(R &&r) const {
    if constexpr (has_local_method<std::ranges::iterator_t<R>>) {
      return std::span(std::ranges::begin(r).local(), std::ranges::size(r));
    } else if constexpr (std::contiguous_iterator<std::ranges::iterator_t<R>>) {
      return std::span(std::ranges::begin(r), std::ranges::size(r));
    }
  }
};

} // namespace

const inline auto local = local_fn_{};

} // namespace ranges

} // namespace lib
