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
