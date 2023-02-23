// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

namespace mhp {

// Select segments local to this rank and convert the iterators in the
// segment to local
auto local_segments(auto &&x) {
  const auto &segments = lib::ranges::segments(x);
  auto is_local = [](const auto &segment) {
    return lib::ranges::rank(segment) == std::size_t(default_comm().rank());
  };
  auto local_iter = [](auto &&segment) {
    auto b = lib::ranges::local(segment.begin());
    auto size = segment.end() - segment.begin();
    return rng::subrange(b, b + size);
  };
  return segments | rng::views::filter(is_local) |
         rng::views::transform(local_iter);
}

} // namespace mhp
