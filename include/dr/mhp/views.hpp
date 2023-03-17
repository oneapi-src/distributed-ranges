// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

namespace mhp {

// Select segments local to this rank and convert the iterators in the
// segment to local
auto local_segments(auto &&dr) {
  auto is_local = [](const auto &segment) {
    return lib::ranges::rank(segment) == default_comm().rank();
  };
  // Convert from remote iter to local iter
  auto local_iter = [](auto &&segment) {
    auto b = lib::ranges::local(segment.begin());
    auto size = segment.end() - segment.begin();
    return rng::subrange(b, b + size);
  };
  return lib::ranges::segments(dr) | rng::views::filter(is_local) |
         rng::views::transform(local_iter);
}

// return locally stored elements as a vector
auto local_vector(auto &&dr) {
  int _i = dr;
  auto lvector = rng::views::zip(local_segments(dr));
  return std::get<0>(*lvector.begin());
}

} // namespace mhp
