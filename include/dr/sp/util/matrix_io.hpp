// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <algorithm>
#include <cassert>
#include <fstream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <dr/views/csr_matrix_view.hpp>
#include <dr/detail/matrix_io.hpp>

namespace dr::sp {

template <typename T, typename I>
auto create_distributed(dr::views::csr_matrix_view<T, I> local_mat,
                        const matrix_partition &partition) {
  dr::sp::sparse_matrix<T, I> a(local_mat.shape(), partition);

  std::vector<dr::views::csr_matrix_view<T, I>> views;
  std::vector<sycl::event> events;
  views.reserve(a.grid_shape()[0] * a.grid_shape()[1]);

  for (I i = 0; i < a.grid_shape()[0]; i++) {
    for (I j = 0; j < a.grid_shape()[1]; j++) {
      auto &&tile = a.tile({i, j});
      dr::index<I> row_bounds(i * a.tile_shape()[0],
                              i * a.tile_shape()[0] + tile.shape()[0]);
      dr::index<I> column_bounds(j * a.tile_shape()[1],
                                 j * a.tile_shape()[1] + tile.shape()[1]);

      auto local_submat = local_mat.submatrix(row_bounds, column_bounds);

      auto submatrix_shape = dr::index<I>(row_bounds[1] - row_bounds[0],
                                          column_bounds[1] - column_bounds[0]);

      auto copied_submat = dr::__detail::convert_to_csr(
          local_submat, submatrix_shape, rng::distance(local_submat),
          std::allocator<T>{});

      auto e = a.copy_tile_async({i, j}, copied_submat);

      views.push_back(copied_submat);
      events.push_back(e);
    }
  }
  __detail::wait(events);

  for (auto &&view : views) {
    dr::__detail::destroy_csr_matrix_view(view, std::allocator<T>{});
  }

  return a;
}


template <typename T, typename I>
auto create_distributed(dr::views::csr_matrix_view<T, I> local_mat) {
  return create_distributed(local_mat, 
      dr::sp::block_cyclic({dr::sp::tile::div, dr::sp::tile::div},
                           {dr::sp::nprocs(), 1}));
}

template <typename T, typename I = std::size_t>
auto mmread(std::string file_path, const matrix_partition &partition,
            bool one_indexed = true) {
  auto local_mat = read_csr<T>(file_path, one_indexed);

  auto a = create_distributed(local_mat, partition);

  dr::__detail::destroy_csr_matrix_view(local_mat, std::allocator<T>{});

  return a;
}

template <typename T, typename I = std::size_t>
auto mmread(std::string file_path, bool one_indexed = true) {
  return mmread<T, I>(
      file_path,
      dr::sp::block_cyclic({dr::sp::tile::div, dr::sp::tile::div},
                           {dr::sp::nprocs(), 1}),
      one_indexed);
}

} // namespace dr::sp
