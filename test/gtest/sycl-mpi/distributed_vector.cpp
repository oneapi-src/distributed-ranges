// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "sycl-mpi-tests.hpp"

using DV = lib::distributed_vector<int, lib::sycl_shared_allocator<int>>;

TEST(SyclMpiTests, DistributedVector) {
  sycl::queue q;
  DV::allocator_type alloc(q);

  DV dv(alloc, 10);
  dv.fence();

  auto p = dv.local().data();
  q.single_task([p]() { *p = 1; }).wait();
  EXPECT_EQ(dv[0], 1);

  dv.fence();
}

TEST(CpuMpiTests, DistributedVectorStencil) {
  DV::allocator_type alloc;

  std::size_t radius = 2;
  lib::halo_bounds hb(radius);

  std::size_t slice = 4;
  std::size_t n = comm_size * slice + 2 * radius;
  DV dv(hb, alloc, n);
  dv.fence();

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

      rng::copy(dv, tv.begin());
      fmt::print("dist:  {}\n", tv);
      EXPECT_EQ(dv.local()[i + radius], dv[comm_rank * slice + i + radius]);
      break;
    }
  }

  dv.fence();
}
