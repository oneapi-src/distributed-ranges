// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/detail/mdspan_shim.hpp>
#include <dr/detail/ranges_shim.hpp>
#include <dr/mhp/containers/distributed_mdarray.hpp>

namespace dr::mhp {

// Transpose mdspan_view. The src is used for temporary storage and is
// undefined after the transpose completes.
template <dr::distributed_mdspan_range MR1, dr::distributed_mdspan_range MR2>
void transpose(MR1 &&src, MR2 &&dst) {
  using T = rng::range_value_t<MR1>;
  assert(!use_sycl());

  constexpr std::size_t rank1 = std::remove_cvref_t<MR1>::rank();
  constexpr std::size_t rank2 = std::remove_cvref_t<MR2>::rank();
  static_assert(rank1 == rank2);

  using index_type = dr::__detail::dr_extents<rank1>;

  // Data decomposition on leading dimension only
  for (std::size_t i = 1; i < rank1; i++) {
    assert(src.grid().extent(i) == 1);
  }

  if constexpr (rank1 == 2) {
    // 2d mdspan

    // swap dimensions of the src to create the dst
    assert(src.mdspan().extent(0) == dst.mdspan().extent(1) &&
           src.mdspan().extent(1) == dst.mdspan().extent(0));

    auto src_tile = src.grid()(default_comm().rank(), 0).mdspan();

    // Divide src tile into sub-tiles by taking vertical slices, each
    // sub-tile is sent to a different rank. The sub-tile is transposed
    // so the number of columns must match the number of rows in the
    // dst tile

    // The alltoall assumes all the ranks have equal size data. The
    // last rank may hold less data, but the actual storage size is
    // uniform.
    std::size_t dst_size = dst.grid()(0, 0).reserved();
    std::size_t src_size = src.grid()(0, 0).reserved();
    std::size_t sub_tile_size = src.grid()(0, 0).mdspan().extent(0) *
                                dst.grid()(0, 0).mdspan().extent(0);
    std::size_t sub_tiles_size = sub_tile_size * default_comm().size();
    dr::drlog.debug(dr::logger::transpose, "sub_tile_size: {}x{}  total: {}\n",
                    src.grid()(0, 0).mdspan().extent(0),
                    dst.grid()(0, 0).mdspan().extent(0), sub_tile_size);

    // The sub-tiles must be contiguous before sending. Use the dst as
    // temporary storage.
    T *tmp_send_buffer = nullptr;
    T *send_buffer =
        dst.grid()(default_comm().rank(), 0).mdspan().data_handle();
    if (dst_size < sub_tiles_size) {
      dr::drlog.debug(
          dr::logger::transpose,
          "Allocating a temporary send buffer dst_size {} sub_tiles_size {}\n",
          dst_size, sub_tiles_size);
      tmp_send_buffer = __detail::allocator<T>().allocate(sub_tiles_size);
      send_buffer = tmp_send_buffer;
    }
    T *buffer = send_buffer;

    index_type start({0, 0}), end({src_tile.extent(0), 0});
    for (std::size_t i = 0; i < dst.grid().extent(0); i++) {
      auto num_cols = dst.grid()(i, 0).mdspan().extent(0);

      end[1] = start[1] + num_cols;
      dr::drlog.debug(dr::logger::transpose, "Packing start: {}, end: {}\n",
                      start, end);
      auto sub_tile = dr::__detail::make_submdspan(src_tile, start, end);
      dr::__detail::mdtranspose<decltype(sub_tile), 1, 0> sub_tile_t(sub_tile);
      dr::__detail::mdspan_copy(sub_tile_t, buffer);
      buffer += sub_tile_size;
      start[1] += num_cols;
    }

    // We have packed the src into the send_buffer and no longer need
    // it. Reuse its space to receive from the other ranks.
    T *tmp_receive_buffer = nullptr;
    T *receive_buffer =
        src.grid()(default_comm().rank(), 0).mdspan().data_handle();
    if (src_size < sub_tiles_size) {
      dr::drlog.debug(dr::logger::transpose,
                      "Allocating a temporary receive buffer src_size {} "
                      "sub_tiles_size {}\n",
                      src_size, sub_tiles_size);
      tmp_receive_buffer = __detail::allocator<T>().allocate(sub_tiles_size);
      receive_buffer = tmp_receive_buffer;
    }
    buffer = receive_buffer;
    default_comm().alltoall(send_buffer, receive_buffer, sub_tile_size);
    auto dst_tile = dst.grid()(default_comm().rank(), 0).mdspan();

    start = {0, 0};
    end = {dst_tile.extent(0), 0};
    for (std::size_t i = 0; i < src.grid().extent(0); i++) {
      auto num_cols = src.grid()(i, 0).mdspan().extent(0);

      end[1] = start[1] + num_cols;
      dr::drlog.debug(dr::logger::transpose, "Unpacking start: {}, end: {}\n",
                      start, end);
      auto sub_tile = dr::__detail::make_submdspan(dst_tile, start, end);
      dr::__detail::mdspan_copy(buffer, sub_tile);
      buffer += sub_tile_size;
      start[1] += num_cols;
    }
    if (tmp_send_buffer) {
      __detail::allocator<T>().deallocate(tmp_send_buffer, sub_tiles_size);
      tmp_send_buffer = nullptr;
    }
    if (tmp_receive_buffer) {
      __detail::allocator<T>().deallocate(tmp_receive_buffer, sub_tiles_size);
      tmp_receive_buffer = nullptr;
    }

  } else {
    assert(false);
  }
  barrier();
}

} // namespace dr::mhp
