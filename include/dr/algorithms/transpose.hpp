namespace lib {

namespace collective {

template <typename MSRC, typename MDST>
inline void transpose(const MSRC &src, MDST &dst) {
  assert(src.extents().extent(0) == dst.extents().extent(1));
  assert(src.extents().extent(1) == dst.extents().extent(0));
  static_assert(MSRC::extents_type::rank() == 2);
  static_assert(MDST::extents_type::rank() == 2);

  for (std::size_t i = 0; i < src.extents().extent(0); i++) {
    for (std::size_t j = 0; j < src.extents().extent(1); j++) {
      dst(j, i) = src(i, j);
    }
  }
}

} // namespace collective

} // namespace lib
