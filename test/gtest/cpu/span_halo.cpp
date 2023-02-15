// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu-tests.hpp"

using halo = lib::span_halo<int>;
using group = halo::group_type;

const std::size_t n = 10;

int value(int rank, int index) { return (rank + 1) * 100 + index; }

struct stencil_data {
  stencil_data(std::size_t size, lib::halo_bounds hb) {
    initial.resize(n);
    for (std::size_t i = 0; i < n; i++) {
      initial[i] = value(comm_rank, i);
    }
    ref = test = initial;

    auto prev = (comm_rank - 1 + comm_size) % comm_size;
    auto next = (comm_rank + 1) % comm_size;

    if (hb.periodic || comm_rank != 0) {
      std::iota(ref.begin(), ref.begin() + hb.prev,
                value(prev, n - 2 * hb.next));
    }
    if (hb.periodic || comm_rank != comm_size - 1) {
      std::iota(ref.end() - hb.next, ref.end(), value(next, hb.prev));
    }
  }

  void check() { EXPECT_TRUE(unary_check(initial, ref, test)); }

  std::vector<int> initial, test, ref;
};

TEST(CpuMpiTests, SpanHaloPeriodic) {
  int radius = 2;
  bool periodic = true;
  lib::halo_bounds hb(radius, periodic);
  stencil_data sd(n, hb);

  halo h(comm, sd.test, hb);

  h.exchange_begin();
  h.exchange_finalize();

  sd.check();
}

TEST(CpuMpiTests, SpanHaloPeriodicRadius1) {
  int radius = 1;
  bool periodic = true;
  lib::halo_bounds hb(radius, periodic);
  stencil_data sd(n, hb);

  halo h(comm, sd.test, hb);

  h.exchange_begin();
  h.exchange_finalize();

  sd.check();
}

TEST(CpuMpiTests, SpanHaloNonPeriodic) {
  int radius = 2;
  bool periodic = false;
  lib::halo_bounds hb(radius, periodic);
  stencil_data sd(n, hb);

  halo h(comm, sd.test, hb);

  h.exchange_begin();
  h.exchange_finalize();

  sd.check();
}

TEST(CpuMpiTests, SpanHaloPointer) {
  int radius = 2;
  bool periodic = false;
  lib::halo_bounds hb(radius, periodic);
  stencil_data sd(n, hb);

  halo h(comm, sd.test.data(), sd.test.size(), hb);

  h.exchange_begin();
  h.exchange_finalize();

  sd.check();
}

TEST(CpuMpiTests, SpanHaloDistributedVector) {
  std::size_t radius = 2;
  std::size_t slice = 4;
  std::size_t n = comm_size * slice + 2 * radius;
  lib::halo_bounds hb(radius);
  lib::distributed_vector<int> dv(hb, n);

  EXPECT_EQ(dv.local().size(), slice + 2 * radius);
  EXPECT_EQ(hb.next, radius);
  EXPECT_EQ(hb.prev, radius);

  if (comm_rank == 0) {
    std::iota(dv.begin(), dv.end(), 1);
  }
  dv.fence();

  for (std::size_t i = 0; i < slice; i++) {
    if (dv.local()[i + radius] != dv[comm_rank * slice + i + radius]) {
      fmt::print("local: {}\n", dv.local());
      fmt::print("dist:  {}\n", dv);
      EXPECT_EQ(dv.local()[i + radius], dv[comm_rank * slice + i + radius]);
      break;
    }
  }
  dv.fence();

  dv.halo().exchange_begin();
  dv.halo().exchange_finalize();

  for (std::size_t i = 0; i < slice + 2 * radius; i++) {
    if (dv.local()[i] != dv[comm_rank * slice + i]) {
      fmt::print("local: {}\n", dv.local());
      std::vector<int> tv(dv.size());
      ;
      rng::copy(dv, tv.begin());
      fmt::print("dist:  {}\n", tv);
      EXPECT_EQ(dv.local()[i + radius], dv[comm_rank * slice + i + radius]);
      break;
    }
  }
}
