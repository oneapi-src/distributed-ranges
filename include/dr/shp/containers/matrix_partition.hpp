// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/shp/containers/detail.hpp>
#include <dr/shp/containers/index.hpp>
#include <dr/shp/init.hpp>

namespace dr::shp {

namespace tile {

// Special constant to indicate tile dimensions of
// {ceil(m / p_m), ceil(n / p_n)} should be chosen
// in order to evenly divide a dimension amongst the
// ranks in the processor grid.
inline constexpr std::size_t div = std::numeric_limits<std::size_t>::max();

} // namespace tile

class matrix_partition {
public:
  virtual std::size_t tile_rank(dr::shp::index<> matrix_shape,
                                dr::shp::index<> tile_id) const = 0;
  virtual dr::shp::index<> grid_shape(dr::shp::index<> matrix_shape) const = 0;
  virtual dr::shp::index<> tile_shape(dr::shp::index<> matrix_shape) const = 0;

  virtual std::unique_ptr<matrix_partition> clone() const = 0;
  virtual ~matrix_partition(){};
};

class block_cyclic final : public matrix_partition {
public:
  block_cyclic(dr::shp::index<> tile_shape = {dr::shp::tile::div,
                                              dr::shp::tile::div},
               dr::shp::index<> grid_shape = detail::factor(dr::shp::nprocs()))
      : tile_shape_(tile_shape), grid_shape_(grid_shape) {}

  block_cyclic(const block_cyclic &) noexcept = default;

  dr::shp::index<> tile_shape() const { return tile_shape_; }

  std::size_t tile_rank(dr::shp::index<> matrix_shape,
                        dr::shp::index<> tile_id) const {
    dr::shp::index<> pgrid_idx = {tile_id[0] % grid_shape_[0],
                                  tile_id[1] % grid_shape_[1]};

    auto pgrid = processor_grid_();

    return pgrid[pgrid_idx[0] * grid_shape_[1] + pgrid_idx[1]];
  }

  dr::shp::index<> grid_shape(dr::shp::index<> matrix_shape) const {
    auto ts = this->tile_shape(matrix_shape);

    return dr::shp::index<>((matrix_shape[0] + ts[0] - 1) / ts[0],
                            (matrix_shape[1] + ts[1] - 1) / ts[1]);
  }

  dr::shp::index<> tile_shape(dr::shp::index<> matrix_shape) const {
    std::array<std::size_t, 2> tshape = {tile_shape_[0], tile_shape_[1]};

    constexpr std::size_t ndims = 2;
    for (std::size_t i = 0; i < ndims; i++) {
      if (tshape[i] == dr::shp::tile::div) {
        tshape[i] = (matrix_shape[i] + grid_shape_[i] - 1) / grid_shape_[i];
      }
    }

    return tshape;
  }

  std::unique_ptr<matrix_partition> clone() const noexcept {
    return std::unique_ptr<matrix_partition>(new block_cyclic(*this));
  }

private:
  std::vector<std::size_t> processor_grid_() const {
    std::vector<std::size_t> grid(grid_shape_[0] * grid_shape_[1]);

    for (std::size_t i = 0; i < grid.size(); i++) {
      grid[i] = i;
    }
    return grid;
  }

  dr::shp::index<> tile_shape_;
  dr::shp::index<> grid_shape_;
};

} // namespace dr::shp
