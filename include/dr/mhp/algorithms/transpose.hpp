// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/detail/mdspan_shim.hpp>
#include <dr/detail/ranges_shim.hpp>
#include <dr/mhp/containers/distributed_mdarray.hpp>

namespace dr::mhp::__detail {

template <typename T> class tmp_buffer {
public:
  tmp_buffer(std::size_t size, auto &&candidate) {
    // Try to use the candidate for storage
    data_ = candidate.mdspan().data_handle();
    size_ = size;
    allocated_data_ = nullptr;

    // Allocate a temporary buffer if it is too small
    if (size_ > candidate.reserved()) {
      dr::drlog.debug(
          dr::logger::transpose,
          "Allocating a temporary buffer requested size {} candidate size {}\n",
          size, candidate.reserved());
      allocated_data_ = __detail::allocator<T>().allocate(size_);
      data_ = allocated_data_;
    }
    assert(data_ != nullptr);
  }

  T *data() { return data_; }

  ~tmp_buffer() {
    // release temporary storage
    if (allocated_data_) {
      __detail::allocator<T>().deallocate(allocated_data_, size_);
      allocated_data_ = nullptr;
    }
  }

private:
  T *data_;
  T *allocated_data_ = nullptr;
  std::size_t size_;
};

template <dr::distributed_mdspan_range MR1, dr::distributed_mdspan_range MR2>
void transpose2D(MR1 &&src, MR2 &&dst, auto sm, auto dm) {
  auto comm = default_comm();

  using T = rng::range_value_t<MR1>;

  using index_type = dr::__detail::dr_extents<2>;

  // swap dimensions of the src to create the dst
  assert(sm.extent(0) == dm.extent(1) && sm.extent(1) == dm.extent(0));

  auto src_tile = src.grid()(comm.rank(), 0);
  auto dst_tile = dst.grid()(comm.rank(), 0);

  if (comm.size() == 1) {
    dr::drlog.debug(dr::logger::transpose, "direct transpose on single rank\n");
    auto sm = src_tile.mdspan();
    auto dm = dst_tile.mdspan();
    dr::__detail::mdtranspose<decltype(sm), 1, 0> src_tile_t(sm);
    dr::__detail::mdspan_copy(src_tile_t, dm);

  } else {
    // Divide src tile into sub-tiles by taking vertical slices, each
    // sub-tile is sent to a different rank. The sub-tile is transposed
    // so the number of columns must match the number of rows in the
    // dst tile

    // The alltoall assumes all the ranks have equal size data. The
    // last rank may hold less data, but the actual storage size is
    // uniform.
    std::size_t sub_tile_size = src.grid()(0, 0).mdspan().extent(0) *
                                dst.grid()(0, 0).mdspan().extent(0);
    std::size_t sub_tiles_size = sub_tile_size * comm.size();
    dr::drlog.debug(dr::logger::transpose, "sub_tile_size: {}x{}  total: {}\n",
                    src.grid()(0, 0).mdspan().extent(0),
                    dst.grid()(0, 0).mdspan().extent(0), sub_tile_size);

    // create a send buffer. try to reuse destination for storage
    __detail::tmp_buffer<T> send_buffer(sub_tiles_size, dst_tile);
    T *buffer = send_buffer.data();

    index_type start({0, 0}), end({src_tile.mdspan().extent(0), 0});
    for (std::size_t i = 0; i < dst.grid().extent(0); i++) {
      auto num_cols = dst.grid()(i, 0).mdspan().extent(0);

      end[1] = start[1] + num_cols;
      dr::drlog.debug(dr::logger::transpose, "Packing start: {}, end: {}\n",
                      start, end);
      auto sub_tile =
          dr::__detail::make_submdspan(src_tile.mdspan(), start, end);
      dr::__detail::mdtranspose<decltype(sub_tile), 1, 0> sub_tile_t(sub_tile);
      dr::__detail::mdspan_copy(sub_tile_t, buffer);
      buffer += sub_tile_size;
      start[1] += num_cols;
    }

    // We have packed the src into the send_buffer and no longer need
    // it. Try to reuse its space for the receive buffer
    __detail::tmp_buffer<T> receive_buffer(sub_tiles_size, src_tile);
    buffer = receive_buffer.data();
    comm.alltoall(send_buffer.data(), receive_buffer.data(), sub_tile_size);

    start = {0, 0};
    end = {dst_tile.mdspan().extent(0), 0};
    for (std::size_t i = 0; i < src.grid().extent(0); i++) {
      auto num_cols = src.grid()(i, 0).mdspan().extent(0);

      end[1] = start[1] + num_cols;
      dr::drlog.debug(dr::logger::transpose, "Unpacking start: {}, end: {}\n",
                      start, end);
      auto sub_tile =
          dr::__detail::make_submdspan(dst_tile.mdspan(), start, end);
      dr::__detail::mdspan_copy(buffer, sub_tile);
      buffer += sub_tile_size;
      start[1] += num_cols;
    }
  }
}

template <dr::distributed_mdspan_range MR1, dr::distributed_mdspan_range MR2>
void transpose3D(MR1 &&src, MR2 &&dst, auto sm, auto dm) {
  auto comm = default_comm();

  using T = rng::range_value_t<MR1>;

  using index_type = dr::__detail::dr_extents<3>;
  // 3d mdspan
  // The transpose is needed to make the first dimension contiguous.
  //
  // i, j, k to j, k, i
  dr::drlog.debug(dr::logger::transpose,
                  "transpose src: [{}, {}, {}]  dst: [{}, {}, {}]\n",
                  sm.extent(0), sm.extent(1), sm.extent(2), dm.extent(0),
                  dm.extent(1), dm.extent(2));

  assert(sm.extent(0) == dm.extent(2) && sm.extent(1) == dm.extent(0) &&
         sm.extent(2) == dm.extent(1));

  auto origin_dst_tile = dst.grid()(0, 0, 0).mdspan();
  auto origin_src_tile = src.grid()(0, 0, 0).mdspan();
  std::size_t sub_tile_size = origin_src_tile.extent(0) *
                              origin_dst_tile.extent(0) *
                              origin_src_tile.extent(1);
  std::size_t sub_tiles_size = sub_tile_size * comm.size();

  auto src_tile = src.grid()(comm.rank(), 0, 0);
  auto dst_tile = dst.grid()(comm.rank(), 0, 0);

  if (comm.size() == 1) {
    dr::drlog.debug(dr::logger::transpose, "direct transpose on single rank\n");
    auto sm = src_tile.mdspan();
    auto dm = dst_tile.mdspan();
    dr::__detail::mdtranspose<decltype(sm), 2, 0, 1> src_tile_t(sm);
    dr::__detail::mdspan_copy(src_tile_t, dm);

  } else {

    // create a send buffer. try to reuse destination for storage
    __detail::tmp_buffer<T> send_buffer(sub_tiles_size, dst_tile);

    T *buffer = send_buffer.data();

    index_type start({0, 0, 0}),
        end({src_tile.mdspan().extent(0), 0, src_tile.mdspan().extent(2)});
    for (std::size_t i = 0; i < dst.grid().extent(0); i++) {
      auto num_cols = dst.grid()(i, 0, 0).mdspan().extent(0);

      end[1] = start[1] + num_cols;
      dr::drlog.debug(dr::logger::transpose, "Packing start: {}, end: {}\n",
                      start, end);
      auto sub_tile =
          dr::__detail::make_submdspan(src_tile.mdspan(), start, end);
      dr::__detail::mdtranspose<decltype(sub_tile), 2, 0, 1> sub_tile_t(
          sub_tile);
      dr::drlog.debug(dr::logger::transpose, "subtile_t\n{}\n", sub_tile_t);

      dr::__detail::mdspan_copy(sub_tile_t, buffer);
      buffer += sub_tile_size;
      start[1] += num_cols;
    }

    __detail::tmp_buffer<T> receive_buffer(sub_tiles_size, src_tile);
    buffer = receive_buffer.data();
    comm.alltoall(send_buffer.data(), receive_buffer.data(), sub_tile_size);

    start = {0, 0, 0};
    end = {dst_tile.mdspan().extent(0), dst_tile.mdspan().extent(1), 0};
    for (std::size_t i = 0; i < src.grid().extent(0); i++) {
      auto num_cols = src.grid()(i, 0, 0).mdspan().extent(0);

      end[2] = start[2] + num_cols;
      dr::drlog.debug(dr::logger::transpose, "Unpacking start: {}, end: {}\n",
                      start, end);
      auto sub_tile =
          dr::__detail::make_submdspan(dst_tile.mdspan(), start, end);
      dr::__detail::mdspan_copy(buffer, sub_tile);
      buffer += sub_tile_size;
      start[2] += num_cols;
    }
  }
}

}; // namespace dr::mhp::__detail

namespace dr::mhp {

// Transpose mdspan_view. The src is used for temporary storage and is
// undefined after the transpose completes.
template <dr::distributed_mdspan_range MR1, dr::distributed_mdspan_range MR2>
void transpose(MR1 &&src, MR2 &&dst) {
  constexpr std::size_t rank1 = std::remove_cvref_t<MR1>::rank();
  constexpr std::size_t rank2 = std::remove_cvref_t<MR2>::rank();
  static_assert(rank1 == rank2);

  // Data decomposition on leading dimension only
  for (std::size_t i = 1; i < rank1; i++) {
    assert(src.grid().extent(i) == 1);
  }

  auto sm = src.mdspan();
  auto dm = dst.mdspan();

  if constexpr (rank1 == 2) {
    __detail::transpose2D(src, dst, sm, dm);
  } else if constexpr (rank1 == 3) {
    __detail::transpose3D(src, dst, sm, dm);
  } else {
    assert(false);
  }
  barrier();
}

} // namespace dr::mhp
