#pragma once

#include <ranges>
#include <shp/zip_view.hpp>

namespace shp {

template <std::ranges::range R> auto enumerate(R &&r) {
  auto i = std::ranges::views::iota(int32_t(0), int32_t(std::ranges::size(r)));
  return shp::zip_view(i, r);
}

} // namespace shp
