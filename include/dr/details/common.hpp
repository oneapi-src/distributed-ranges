// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

namespace lib {

template <typename T> T product(T first) { return first; }

template <typename T, typename... Args> T product(T first, Args... args) {
  return first * product(args...);
}

// Equal size partitions, exactly covers num
inline std::size_t partition(std::size_t num, std::size_t denom) {
  assert(num % denom == 0);
  return num / denom;
}

// Equal size partitions, rounding up to cover num
inline std::size_t partition_up(std::size_t num, std::size_t multiple) {
  return (num + multiple - 1) / multiple;
}

inline std::size_t round_up(std::size_t num, std::size_t multiple) {
  return partition_up(num, multiple) * multiple;
}

inline std::size_t storage_size(std::size_t dim, std::size_t comm_size) {
  return round_up(dim, comm_size);
}

inline std::size_t storage_size(const auto &extents, std::size_t comm_size) {
  // Assume partition on leading dimension. Leading dimension must
  // divide evenly
  std::size_t size = storage_size(extents.extent(0), comm_size);
  for (std::size_t i = 1; i < extents.rank(); i++) {
    size *= extents.extent(i);
  }
  return size;
}

template <class F, class Tuple>
constexpr void tuple_foreach(F &&f, Tuple &&tuple) {
  std::apply(
      [&]<class... Ts>(Ts &&...elements) {
        (std::invoke(f, std::forward<Ts>(elements)), ...);
      },
      std::forward<Tuple>(tuple));
}

} // namespace lib
