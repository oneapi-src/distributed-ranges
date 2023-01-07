// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

namespace lib {

template <typename R> class view_local_span : public rng::view_base {
  using distributed_container_type =
      typename rng::iterator_t<R>::container_type;
  using local_container_type = typename distributed_container_type::local_type;
  using local_iterator = typename rng::iterator_t<local_container_type>;

public:
  /// Constructor
  view_local_span(R &&r) : begin_(r.begin().local()), end_(r.end().local()) {}

  local_iterator begin() { return begin_; }
  local_iterator end() { return end_; }

private:
  local_iterator begin_;
  local_iterator end_;
};

template <typename R> view_local_span(R &&) -> view_local_span<R>;

struct local_span {
  template <typename R> auto operator()(R &&r) const {
    return view_local_span(std::forward<R>(r));
  }

  template <typename R> friend auto operator|(R &&r, const local_span &) {
    return view_local_span(std::forward<R>(r));
  }
};

template <typename... Iters> struct zip_iterator {
private:
  static auto zip_iterators(Iters... iters)
    requires(sizeof...(Iters) == 2)
  {
    return std::pair(iters...);
  }
  static auto zip_iterators(Iters... iters)
    requires(sizeof...(Iters) > 2)
  {
    return std::tuple(iters...);
  }

  static auto zip_reference(Iters... iters)
    requires(sizeof...(Iters) == 2)
  {
    return std::pair<std::iter_reference_t<Iters>...>(*iters...);
  }
  static auto zip_reference(Iters... iters)
    requires(sizeof...(Iters) > 2)
  {
    return std::tuple<std::iter_reference_t<Iters>...>(*iters...);
  }

  static auto zip_value(Iters... iters)
    requires(sizeof...(Iters) == 2)
  {
    return std::pair<std::iter_value_t<Iters>...>(*iters...);
  }
  static auto zip_value(Iters... iters)
    requires(sizeof...(Iters) > 2)
  {
    return std::tuple<std::iter_value_t<Iters>...>(*iters...);
  }

public:
  using iterator_category = std::random_access_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using value_type = decltype(zip_value(std::declval<Iters>()...));
  using reference = decltype(zip_reference(std::declval<Iters>()...));

  zip_iterator(Iters... iters) : iters_(iters...) {}
  // Sentinel that never matches
  zip_iterator() : iters_() {}

  // Iterator to reference
  auto operator*() const {
    auto ref = [](Iters... iters) { return zip_reference(iters...); };
    return std::apply(ref, iters_);
  }
  auto operator[](difference_type n) const { return *(*this + n); }

  // Comparison
  bool operator==(const zip_iterator other) const {
    return std::get<0>(iters_) == std::get<0>(other.iters_);
  }
  auto operator<=>(const zip_iterator &other) const {
    return std::get<0>(iters_) <=> std::get<0>(other.iters_);
  }

  // These are the only arithmetic operators that do elementwise
  // operations
  zip_iterator &operator-=(difference_type n) {
    tuple_foreach([=](auto &it) { it -= n; }, iters_);
    return *this;
  }
  zip_iterator &operator+=(difference_type n) {
    tuple_foreach([=](auto &it) { it += n; }, iters_);
    return *this;
  }

  // prefix
  zip_iterator &operator++() {
    *this += 1;
    return *this;
  }
  zip_iterator &operator--() {
    *this -= 1;
    return *this;
  }

  // postfix
  zip_iterator operator++(int) {
    auto prev = *this;
    *this += 1;
    return prev;
  }
  zip_iterator operator--(int) {
    auto prev = *this;
    *this -= 1;
    return prev;
  }

  zip_iterator operator+(difference_type n) const {
    zip_iterator it(*this);
    it += n;
    return it;
  }
  zip_iterator operator-(difference_type n) const {
    zip_iterator it(*this);
    it -= n;
    return it;
  }

  // non member operators when zip_iterator is not the first operand
  friend zip_iterator operator+(difference_type n, const zip_iterator other) {
    return n + other;
  }
  friend zip_iterator operator-(difference_type n, const zip_iterator other) {
    return n - other;
  }

private:
  using iters_type = decltype(zip_iterators(std::declval<Iters>()...));
  iters_type iters_;
};

template <rng::input_range... Views> class zip_view : public rng::view_base {

public:
  /// iterator type
  using iterator = zip_iterator<rng::iterator_t<Views>...>;

  // Convert ranges::unreachable_sentinel_t, to
  // rng::iterator_t<Views>{}. Do this with an overloaded conversion
  // function that is otherwise identity Pass target conversion type
  // as template argument rng::iterator_t<Views>

  /// Constructor
  zip_view(Views &...views)
      : begin_(views.begin()...), end_(convert_end(views)...) {}

  /// Begin
  iterator begin() const { return begin_; }
  /// End
  iterator end() const { return end_; }
  /// [] operator
  auto operator[](std::size_t n) const { return *(begin() + n); }

private:
  static auto convert_end(auto &&view) {
    using end_type = decltype(view.end());
    if constexpr (std::is_same_v<rng::unreachable_sentinel_t, end_type>) {
      return rng::iterator_t<decltype(view)>{};
    } else {
      return view.end();
    }
  }

  iterator begin_;
  iterator end_;
};

template <typename... Rng>
zip_view(Rng &&...) -> zip_view<rng::views::all_t<Rng>...>;

// Return an array of pointers given a tuple of refs
template <typename Tuple> constexpr auto refs_to_pointers(Tuple &&ref_tuple) {
  constexpr auto pointer_array = [](auto &&...x) {
    return std::array{(&x)...};
  };
  return std::apply(pointer_array, std::forward<Tuple>(ref_tuple));
}

// Return an array of begin iterators given a zip range
inline auto begin_iterators(auto zr) { return refs_to_pointers(*zr.begin()); }

// Return an array of end iterators given a zip range
inline auto end_iterators(auto zr) { return refs_to_pointers(*zr.end()); }

// Return true if all the ranges in a zip range are conformant
inline bool zip_range_conformant(auto zr) {
  const auto &its = begin_iterators(zr);
  for (std::size_t i = 1; i < its.size(); i++) {
    if (!its[0].conforms(its[i]))
      return false;
  }
  return true;
}

// Perform a fence on all the containers in a zip range
inline void zip_range_fence(auto zr) {
  for (auto &it : begin_iterators(zr)) {
    it.container().fence();
  }
}

inline auto zip_range_comm(auto zr) {
  return (&std::get<0>(*zr.begin())).container().comm();
}

template <distributed_range_zip ZR>
class view_local_zip_span : public rng::view_base {
private:
  // Return a zip of local ranges, given a zip of global ranges
  static auto local_zipped_range(ZR &&zr) {
    auto b = begin_iterators(zr);
    auto e = end_iterators(zr);
    // We have global begin iterators and global end interators in
    // separate arrays.  Convert them to local ranges, and zip into a
    // single range.
    auto zip_locals = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
      return rng::views::zip(rng::subrange(b[Is].local(), e[Is].local())...);
    };
    return zip_locals(std::make_index_sequence<std::size(b)>());
  }

  decltype(local_zipped_range(std::declval<ZR>())) local_zr_;

public:
  /// Constructor
  view_local_zip_span(ZR &&zr) : local_zr_(local_zipped_range(zr)) {}

  auto begin() { return local_zr_.begin(); }
  auto end() { return local_zr_.end(); }
};

template <distributed_range_zip R>
view_local_zip_span(R &&) -> view_local_zip_span<R>;

struct local_zip_span {
  template <distributed_range_zip R> auto operator()(R &&r) const {
    return view_local_zip_span(std::forward<R>(r));
  }

  template <distributed_range_zip R>
  friend auto operator|(R &&r, const local_zip_span &) {
    return view_local_zip_span(std::forward<R>(r));
  }
};

} // namespace lib
