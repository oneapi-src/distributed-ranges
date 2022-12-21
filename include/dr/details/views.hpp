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

} // namespace lib
