// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/detail/mdspan_shim.hpp>
#include <dr/detail/ranges_shim.hpp>

namespace dr::mhp::__detail {

// Divide a tile into sub-tiles by partitioning the columns. Sub-tiles
// are not contiguous and sent to remote ranks, so they need to be
// packed.
//
// The last sub-tile may be smaller than the other tiles in both 1st
// (distribution dimension) and 2nd dimension (constrained by 1st
// dimension because of transpose operation) . Using a uniform tile
// size will write beyond the end of a row so we need to loop
// through sub-tiles.
//
// MKL cannot be used because it only supports a limited set of
// types.

void copy_pack(auto &&in, auto &&out, auto &&in_tile, std::size_t subtile_size,
               auto send_buf) {
  for (std::size_t out_index = 0; out_index < out.grid().extent(0);
       out_index++) {
    auto buf = &send_buf[out_index * subtile_size];
    auto out_segment = out.grid()(out_index, 0);
    auto out_tile = out_segment.mdspan();
    std::size_t base = out_segment.origin()[0];
    for (std::size_t i = 0; i < in_tile.extent(0); i++) {
      for (std::size_t j = 0; j < out_tile.extent(0); j++) {
        *buf++ = in_tile(i, base + j);
      }
    }
  }
}

void unpack_transpose(auto &&in, auto &&out, auto &&out_tile,
                      std::size_t subtile_size, auto receive_buf) {
  for (std::size_t in_index = 0; in_index < in.grid().extent(0); in_index++) {
    auto buf = &receive_buf[in_index * subtile_size];
    auto in_segment = in.grid()(in_index, 0);
    auto in_tile = in_segment.mdspan();
    std::size_t col_base = in_segment.origin()[0];
    for (std::size_t i = 0; i < in_tile.extent(0); i++) {
      for (std::size_t j = 0; j < out_tile.extent(0); j++) {
        out_tile(j, col_base + i) = *buf++;
      }
    }
  }
}

} // namespace dr::mhp::__detail

namespace dr::mhp {

// transpose: swap first 2 dimensions of a mdspan_view
void transpose(dr::distributed_mdspan_range auto &&in,
               dr::distributed_mdspan_range auto &&out) {
  // 2d mdspan, with in/out shape for swapping dim 1 & 2
  assert(in.mdspan().rank() == 2 && in.mdspan().rank() == out.mdspan().rank());
  assert(in.mdspan().extent(0) == out.mdspan().extent(1) &&
         in.mdspan().extent(1) == out.mdspan().extent(0));

  // Decomposition along leading dimension
  assert(in.grid().extent(0) == default_comm().size() && // dr-style ignore
         in.grid().extent(1) == 1);
  auto in_tile = in.grid()(default_comm().rank(), 0).mdspan();
  auto out_tile = out.grid()(default_comm().rank(), 0).mdspan();

  // Packed data for send/receive
  using T = rng::range_value_t<decltype(in)>;
  // use top left corner because sub-tiles on right/bottom edge are
  // smaller
  auto in_base = in.grid()(0, 0).mdspan();
  auto out_base = out.grid()(0, 0).mdspan();
  std::size_t subtile_size = in_base.extent(0) * out_base.extent(0);
  std::size_t buffer_size = subtile_size * in.grid().extent(0);
  T *send_buf = __detail::allocator<T>().allocate(buffer_size);
  T *receive_buf = __detail::allocator<T>().allocate(buffer_size);

  __detail::copy_pack(in, out, in_tile, subtile_size, send_buf);

  default_comm().alltoall(send_buf, receive_buf, subtile_size);

  __detail::unpack_transpose(in, out, out_tile, subtile_size, receive_buf);
  __detail::allocator<T>().deallocate(send_buf, buffer_size);
  __detail::allocator<T>().deallocate(receive_buf, buffer_size);
}

} // namespace dr::mhp
