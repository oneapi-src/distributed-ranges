// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

namespace lib {

// returns range: [(rank, element) ...]
auto ranked_view(const lib::distributed_range auto &r) {
  auto rank = [](auto &&v) { return lib::ranges::rank(&v); };
  return rng::views::zip(rng::views::transform(r, rank), r);
}

} // namespace lib
