// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu-fuzz.hpp"

using V = std::vector<int>;
using DV = lib::distributed_vector<int>;

void check_transform(std::size_t n, std::size_t b, std::size_t e) {
  auto op = [](auto n) { return n * n; };
  int iota_base = 100;

  lib::distributed_vector<int> dvi1(n), dvr1(n);
  rng::iota(dvi1, iota_base);
  dvi1.fence();
  lib::transform(dvi1.begin() + b, dvi1.begin() + e, dvr1.begin(), op);
  dvr1.fence();

  lib::distributed_vector<int> dvi2(n), dvr2(n);
  rng::iota(dvi2, iota_base);
  dvi2.fence();

  if (comm_rank == 0) {
    std::transform(dvi2.begin() + b, dvi2.begin() + e, dvr2.begin(), op);

    std::vector<int> v(n), vr(n);
    rng::iota(v, iota_base);
    std::transform(v.begin() + b, v.begin() + e, vr.begin(), op);
    assert(is_equal(vr, dvr1));
    assert(is_equal(vr, dvr2));
  }
}

void check_copy(std::size_t n, std::size_t b, std::size_t e) {

  V v_in(n), v(n), v1(n), v2(n);
  rng::iota(v_in, 100);

  lib::distributed_vector<int> dv_in(n), dv1(n), dv2(n), dv3(n), dv4(n), dv5(n),
      dv6(n), dv7(n);
  lib::iota(dv_in, 100);
  lib::copy(dv_in.begin() + b, dv_in.begin() + e, dv1.begin() + b);
  lib::copy(rng::subrange(dv_in.begin() + b, dv_in.begin() + e),
            dv2.begin() + b);

  lib::copy(0, v_in.begin() + b, v_in.begin() + e, dv4.begin() + b);
  lib::copy(0, rng::subrange(v_in.begin() + b, v_in.begin() + e),
            dv6.begin() + b);
  if (comm_rank == 0) {
    lib::copy(0, &*(v_in.begin() + b), e - b, dv7.begin() + b);
  } else {
    lib::copy(0, nullptr, e - b, dv7.begin() + b);
  }
  lib::copy(0, comm_rank == 0 ? &*(v_in.begin() + b) : nullptr, e - b,
            dv5.begin() + b);

  lib::copy(0, dv_in.begin() + b, dv_in.begin() + e, v1.begin() + b);
  lib::copy(0, dv_in.begin() + b, e - b,
            comm_rank == 0 ? &*(v2.begin() + b) : nullptr);

  if (comm_rank == 0) {
    std::copy(dv_in.begin() + b, dv_in.begin() + e, dv3.begin() + b);
  }
  dv3.fence();

  if (comm_rank == 0) {
    std::copy(v_in.begin() + b, v_in.begin() + e, v.begin() + b);
    assert(is_equal(dv1, v));
    assert(is_equal(dv2, v));
    assert(is_equal(dv3, v));
    assert(is_equal(dv4, v));
    assert(is_equal(dv5, v));
    assert(is_equal(dv6, v));
    assert(is_equal(dv7, v));

    assert(is_equal(v1, v));
    assert(is_equal(v2, v));
  }
}
