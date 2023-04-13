// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

namespace mhp {

// Select segments local to this rank and convert the iterators in the
// segment to local
template <typename R> auto local_segments(R &&dr) {
  auto is_local = [](const auto &segment) {
    return lib::ranges::rank(segment) == default_comm().rank();
  };
  // Convert from remote iter to local iter
  auto local_iter = [](const auto &segment) {
    auto b = lib::ranges::local(rng::begin(segment));
    return rng::subrange(b, b + rng::distance(segment));
  };
  return lib::ranges::segments(std::forward<R>(dr)) |
         rng::views::filter(is_local) | rng::views::transform(local_iter);
}

inline constexpr auto take = rng::views::take;

} // namespace mhp
