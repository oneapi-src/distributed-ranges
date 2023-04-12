// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu-fuzz.hpp"

using V = std::vector<int>;
using DV = dr::mhp::distributed_vector<int>;

void check_transform(std::size_t n, std::size_t b, std::size_t e) {
  auto op = [](auto n) { return n * n; };
  int iota_base = 100;

  DV dvi1(n), dvr1(n, 0);
  dr::mhp::iota(dvi1, iota_base);
  dr::mhp::transform(dvi1.begin() + b, dvi1.begin() + e, dvr1.begin(), op);

  DV dvi2(n), dvr2(n, 0);
  dr::mhp::iota(dvi2, iota_base);

  if (comm_rank == 0) {
    std::transform(dvi2.begin() + b, dvi2.begin() + e, dvr2.begin(), op);

    std::vector<int> v(n, 0), vr(n, 0);
    rng::iota(v, iota_base);
    std::transform(v.begin() + b, v.begin() + e, vr.begin(), op);
    assert(is_equal(vr, dvr1));
    assert(is_equal(vr, dvr2));
  }
}

void check_copy(std::size_t n, std::size_t b, std::size_t e) {

  V v_in(n, 0), v(n, 0);
  rng::iota(v_in, 100);

  DV dv_in(n, 0), dv1(n, 0), dv2(n, 0), dv3(n, 0);
  dr::mhp::iota(dv_in, 100);
  dr::mhp::copy(dv_in.begin() + b, dv_in.begin() + e, dv1.begin() + b);
  dr::mhp::copy(rng::subrange(dv_in.begin() + b, dv_in.begin() + e),
                dv2.begin() + b);

  if (comm_rank == 0) {
    std::copy(dv_in.begin() + b, dv_in.begin() + e, dv3.begin() + b);
  }
  dr::mhp::fence();

  if (comm_rank == 0) {
    std::copy(v_in.begin() + b, v_in.begin() + e, v.begin() + b);
    assert(is_equal(dv1, v));
    assert(is_equal(dv2, v));
    assert(is_equal(dv3, v));
  }
}
