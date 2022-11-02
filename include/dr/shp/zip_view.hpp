#pragma once

#include <oneapi/dpl/iterator>
#include <ranges>

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

  zip_view(Rs... rs) : views_(std::ranges::views::all(rs)...) {
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

private:
  template <std::size_t... Is> auto begin_impl_(std::index_sequence<Is...>) {
    return zip_iterator<std::ranges::iterator_t<Rs>...>(
        std::ranges::begin(std::get<Is>(views_))...);
  }

  std::tuple<std::ranges::views::all_t<Rs>...> views_;
  std::size_t size_;
};

template <typename... Rs> zip_view(Rs &&...rs) -> zip_view<Rs...>;

} // namespace shp
