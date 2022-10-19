namespace lib {

/// Copy from range on root to distributed vector
template <ranges::input_range R>
void copy(lib::collective_root_policy e, R &&src,
          distributed_vector<rng::range_value_t<R>> &dst) {
  dst.scatter(src, e.root());
}

/// Copy from distributed vector to range on root
template <ranges::input_range R>
void copy(lib::collective_root_policy e,
          distributed_vector<rng::range_value_t<R>> &src, R &&dst) {
  src.gather(dst, e.root());
}

} // namespace lib
