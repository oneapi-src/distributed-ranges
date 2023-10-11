// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause
#include <cstddef>

// Arakava C grid object
//
// T points at cell centers
// U points at center of x edges
// V points at center of y edges
// F points at vertices
//
//   |       |       |       |       |
//   f---v---f---v---f---v---f---v---f-
//   |       |       |       |       |
//   u   t   u   t   u   t   u   t   u
//   |       |       |       |       |
//   f---v---f---v---f---v---f---v---f-
struct ArakawaCGrid {
  double xmin, xmax;     // x limits in physical coordinates (U point min/max)
  double ymin, ymax;     // y limits in physical coordinates (V point min/max)
  std::size_t nx, ny;    // number of cells (T points)
  double lx, ly;         // grid size in physical coordinates
  double dx, dy;         // cell size in physical coordinates
  double dx_inv, dy_inv; // reciprocial dx and dy

  ArakawaCGrid(double _xmin, double _xmax, double _ymin, double _ymax,
               std::size_t _nx, std::size_t _ny)
      : xmin(_xmin), xmax(_xmax), ymin(_ymin), ymax(_ymax), nx(_nx), ny(_ny),
        lx(_xmax - _xmin), ly(_ymax - _ymin) {
    dx = lx / nx;
    dy = ly / ny;
    dx_inv = 1.0 / dx;
    dy_inv = 1.0 / dy;
  };
};
