namespace lib {

template <typename R> class view_local_span : public rng::view_base {
  using distributed_container_type = rng::iterator_t<R>::container_type;
  using local_container_type = distributed_container_type::local_type;
  using local_iterator = rng::iterator_t<local_container_type>;

public:
  /// Constructor
  view_local_span(R &&r) {
    auto first = r.begin();
    auto last = r.end();
    container_ = &first.object();
    auto [fo, lo] =
        container_->select_local(first, last, container_->comm().rank());
    first_offset_ = fo;
    last_offset_ = lo;
  }

  /// Iterator for beginning of view
  local_iterator begin() { return container_->local().begin() + first_offset_; }

  /// Iterator for end of view
  local_iterator end() { return container_->local().begin() + last_offset_; }

private:
  distributed_container_type *container_;
  std::size_t first_offset_;
  std::size_t last_offset_;
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
