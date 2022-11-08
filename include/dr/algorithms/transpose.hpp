namespace lib {

template <typename T>
concept mdspan_2d = T::extents_type::rank()
== 2;

template <typename T>
concept mdspan_pointer = std::is_same_v<typename T::container_type,
                                        std::vector<typename T::element_type>>;

template <typename T>
concept mdspan_row_major =
    std::is_same_v<typename T::layout_type, stdex::layout_right>;

template <typename T>
concept mdspan_col_major =
    std::is_same_v<typename T::layout_type, stdex::layout_left>;

template <typename T>
concept mdspan_regular = mdspan_pointer<T> &&
    (mdspan_row_major<T> || mdspan_col_major<T>);

template <typename A, typename B> constexpr inline bool mdspan_same_layout() {
  return std::is_same_v<typename A::layout_type, typename B::layout_type>;
}

namespace collective {

template <typename T> inline auto mkl_layout() {
  if (mdspan_row_major<T>)
    return 'R';
  if (mdspan_col_major<T>)
    return 'C';
  assert(false);
}

template <typename src_type, typename dst_type>
inline void mkl_transpose(const src_type &src, dst_type &dst) {
  drlog.debug(nostd::source_location::current(),
              "MKL transpose: layout: {} rows: {} cols: {} lda: {} ldb: {}\n",
              mkl_layout<src_type>(), src.extents().extent(0),
              src.extents().extent(1), src.stride(0), dst.stride(0));
  mkl_domatcopy(mkl_layout<src_type>(), 'T', src.extents().extent(0),
                src.extents().extent(1), 1.0, src.data(), src.stride(0),
                dst.data(), dst.stride(0));
}

template <mdspan_2d src_type, mdspan_2d dst_type>
inline void transpose(const src_type &src, dst_type &dst) {
  if constexpr (mdspan_regular<src_type> && mdspan_regular<dst_type> &&
                mdspan_same_layout<src_type, dst_type>()) {
    mkl_transpose(src, dst);
  } else {
    drlog.debug(nostd::source_location::current(), "Generic transpose\n");
    // Generic mdspan transpose
    for (std::size_t i = 0; i < src.extents().extent(0); i++) {
      for (std::size_t j = 0; j < src.extents().extent(1); j++) {
        dst(j, i) = src(i, j);
      }
    }
  }
}

} // namespace collective

} // namespace lib
