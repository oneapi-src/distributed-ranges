// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <shp/vector.hpp>

namespace shp {

template <typename T, typename Allocator>
class device_vector : public shp::vector<T, Allocator> {
public:
  constexpr device_vector() noexcept {}

  using base = shp::vector<T, Allocator>;

  using value_type = T;
  using size_type = std::size_t;
  using difference_type = std::size_t;

  constexpr device_vector(size_type count, const Allocator &alloc,
                          size_type rank)
      : rank_(rank), base(count, alloc) {}

  constexpr std::size_t rank() const noexcept { return rank_; }

private:
  std::size_t rank_ = 0;
};

} // namespace shp
