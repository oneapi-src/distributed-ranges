namespace lib {

namespace collective {

/// Copy from range on root to distributed vector
template <ranges::input_range R>
void copy(int root, R &&src, distributed_vector<rng::range_value_t<R>> &dst) {
  dst.scatter(src, root);
}

/// Copy from distributed vector to range on root
template <ranges::input_range R>
void copy(int root, distributed_vector<rng::range_value_t<R>> &src, R &&dst) {
  src.gather(dst, root);
}

} // namespace collective

} // namespace lib
