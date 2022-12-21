// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <oneapi/dpl/iterator>
#include <ranges>

#include <concepts/concepts.hpp>
#include <shp/iterator_adaptor.hpp>

namespace {

template <typename T> struct is_owning_view : std::false_type {};
template <std::ranges::range R>
struct is_owning_view<std::ranges::owning_view<R>> : std::true_type {};

template <typename T>
inline constexpr bool is_owning_view_v = is_owning_view<T>{};

template <typename T> struct is_ref_view : std::false_type {};
template <std::ranges::range R>
struct is_ref_view<std::ranges::ref_view<R>> : std::true_type {};

template <typename T> inline constexpr bool is_ref_view_v = is_ref_view<T>{};

} // namespace

namespace shp {

template <std::random_access_iterator... Iters> class zip_accessor {
public:
  using element_type = std::tuple<std::iter_value_t<Iters>...>;
  using value_type = element_type;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using reference = std::tuple<std::iter_reference_t<Iters>...>;

  using iterator_category = std::random_access_iterator_tag;

  using iterator_accessor = zip_accessor;
  using const_iterator_accessor = iterator_accessor;
  using nonconst_iterator_accessor = iterator_accessor;

  constexpr zip_accessor() noexcept = default;
  constexpr ~zip_accessor() noexcept = default;
  constexpr zip_accessor(const zip_accessor &) noexcept = default;
  constexpr zip_accessor &operator=(const zip_accessor &) noexcept = default;

  constexpr zip_accessor(Iters... iters) : iterators_(iters...) {}

  zip_accessor &operator+=(difference_type offset) {
    auto increment = [&](auto &&iter) { iter += offset; };
    iterators_apply_impl_<0>(increment);
    return *this;
  }

  constexpr bool operator==(const iterator_accessor &other) const noexcept {
    return std::get<0>(iterators_) == std::get<0>(other.iterators_);
  }

  constexpr difference_type
  operator-(const iterator_accessor &other) const noexcept {
    return std::get<0>(iterators_) - std::get<0>(other.iterators_);
  }

  constexpr bool operator<(const iterator_accessor &other) const noexcept {
    return std::get<0>(iterators_) < std::get<0>(other.iterators_);
  }

  constexpr reference operator*() const noexcept {
    return get_impl_(std::make_index_sequence<sizeof...(Iters)>{});
  }

private:
  template <std::size_t... Ints>
  reference get_impl_(std::index_sequence<Ints...>) const noexcept {
    return reference(*std::get<Ints>(iterators_)...);
  }

  template <std::size_t I, typename Fn> void iterators_apply_impl_(Fn &&fn) {
    fn(std::get<I>(iterators_));
    if constexpr (I + 1 < sizeof...(Iters)) {
      iterators_apply_impl_<I + 1>(fn);
    }
  }

  std::tuple<Iters...> iterators_;
};

template <std::random_access_iterator... Iters>
using zip_iterator = lib::iterator_adaptor<zip_accessor<Iters...>>;

template <std::ranges::random_access_range... Rs>
class zip_view : public std::ranges::view_interface<zip_view<Rs...>> {
public:
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  zip_view(Rs... rs)
      : views_(std::ranges::views::all(std::forward<Rs>(rs))...) {
    std::array<std::size_t, sizeof...(Rs)> sizes = {
        std::size_t(std::ranges::size(rs))...};

    // TODO: support zipped views with some ranges shorter than others
    size_ = sizes[0];

    for (auto &&size : sizes) {
      size_ = std::min(size_, size);
    }
  }

  std::size_t size() const noexcept { return size_; }

  auto begin() {
    return begin_impl_(std::make_index_sequence<sizeof...(Rs)>{});
  }

  auto end() { return begin() + size(); }

  auto operator[](std::size_t idx) const { return *(begin() + idx); }

  static constexpr bool num_views = sizeof...(Rs);

  template <std::size_t I> decltype(auto) get_view() {
    auto &&view = std::get<I>(views_);

    if constexpr (is_ref_view_v<std::remove_cvref_t<decltype(view)>> ||
                  is_owning_view_v<std::remove_cvref_t<decltype(view)>>) {
      return view.base();
    } else {
      return view;
    }
  }

  // If there is at least one distributed range, expose segments
  // of overlapping remote ranges.
  auto segments()
    requires(lib::distributed_range<Rs> || ...)
  {
    std::array<std::size_t, sizeof...(Rs)> segment_ids;
    std::array<std::size_t, sizeof...(Rs)> local_idx;
    segment_ids.fill(0);
    local_idx.fill(0);

    constexpr std::size_t num_views = sizeof...(Rs);

    size_t cumulative_size = 0;
    size_t segment_id = 0;

    using segment_view_type = decltype(get_zipped_view_impl_(
        segment_ids, local_idx, 0, std::make_index_sequence<sizeof...(Rs)>{}));
    std::vector<segment_view_type> segment_views;

    while (cumulative_size < size()) {
      auto size = get_next_segment_size(segment_ids, local_idx);

      cumulative_size += size;

      // Create zipped segment with
      // zip_view(segments()[Is].subspan(local_idx[Is], size)...) And some rank
      // (e.g. get_view<0>.rank())
      auto segment_view =
          get_zipped_view_impl_(segment_ids, local_idx, size,
                                std::make_index_sequence<sizeof...(Rs)>{});

      segment_views.push_back(std::move(segment_view));

      increment_local_idx(segment_ids, local_idx, size);
    }

    return segment_views;
  }

  // If:
  //   - There is at least one remote range in the zip
  //   - There are no distributed ranges in the zip
  // Expose a rank.
  std::size_t rank()
    requires((lib::remote_range<Rs> || ...) &&
             !(lib::distributed_range<Rs> || ...))
  {
    return get_rank_impl_<0, Rs...>();
  }

private:
  template <std::size_t I, typename R> std::size_t get_rank_impl_() {
    static_assert(I < sizeof...(Rs));
    return lib::ranges::rank(get_view<I>());
  }

  template <std::size_t I, typename R, typename... Rs_>
    requires(sizeof...(Rs_) > 0)
  std::size_t get_rank_impl_() {
    static_assert(I < sizeof...(Rs));
    if constexpr (lib::remote_range<R>) {
      return lib::ranges::rank(get_view<I>());
    } else {
      return get_rank_impl_<I + 1, Rs_...>();
    }
  }

  template <typename T> auto create_view_impl_(T &&t) {
    if constexpr (lib::remote_range<T>) {
      return shp::device_span(std::forward<T>(t));
    } else {
      return shp::span(std::forward<T>(t));
    }
  }

  template <std::size_t... Is>
  auto get_zipped_view_impl_(auto &&segment_ids, auto &&local_idx,
                             std::size_t size, std::index_sequence<Is...>) {
    return zip_view<decltype(create_view_impl_(
                                 segment_or_orig_(get_view<Is>(),
                                                  segment_ids[Is]))
                                 .subspan(local_idx[Is], size))...>(
        create_view_impl_(segment_or_orig_(get_view<Is>(), segment_ids[Is]))
            .subspan(local_idx[Is], size)...);
  }

  template <std::size_t I = 0>
  void increment_local_idx(auto &&segment_ids, auto &&local_idx,
                           std::size_t size) {
    local_idx[I] += size;

    if (local_idx[I] >=
        std::ranges::size(segment_or_orig_(get_view<I>(), segment_ids[I]))) {
      local_idx[I] = 0;
      segment_ids[I]++;
    }

    if constexpr (I + 1 < sizeof...(Rs)) {
      increment_local_idx<I + 1>(segment_ids, local_idx, size);
    }
  }

  template <std::size_t... Is> auto begin_impl_(std::index_sequence<Is...>) {
    return zip_iterator<std::ranges::iterator_t<Rs>...>(
        std::ranges::begin(std::get<Is>(views_))...);
  }

  template <typename T, typename U> T min_many_impl_(T t, U u) {
    if (u < t) {
      return u;
    } else {
      return t;
    }
  }

  template <lib::distributed_range T>
  decltype(auto) segment_or_orig_(T &&t, std::size_t idx) {
    return lib::ranges::segments(t)[idx];
  }

  template <typename T>
  decltype(auto) segment_or_orig_(T &&t, std::size_t idx) {
    return t;
  }

  template <typename T, typename U, typename... Ts>
  T min_many_impl_(T t, U u, Ts... ts) {
    T local_min = min_many_impl_(t, u);
    return min_many_impl_(local_min, ts...);
  }

  template <std::size_t... Is>
  std::size_t get_next_segment_size_impl_(auto &&segment_ids, auto &&local_idx,
                                          std::index_sequence<Is...>) {
    return min_many_impl_(
        std::ranges::size(segment_or_orig_(get_view<Is>(), segment_ids[Is])) -
        local_idx[Is]...);
  }

  std::size_t get_next_segment_size(auto &&segment_ids, auto &&local_idx) {
    return get_next_segment_size_impl_(
        segment_ids, local_idx, std::make_index_sequence<sizeof...(Rs)>{});
  }

  std::tuple<std::ranges::views::all_t<Rs>...> views_;
  std::size_t size_;
};

template <typename... Rs> zip_view(Rs &&...rs) -> zip_view<Rs...>;

} // namespace shp
