#include "cpu-mpi-tests.hpp"

TEST(CpuMpiTests, HaloSegment) {
  const int n = 10;
  std::vector<int> test_data(n);
  std::iota(test_data.begin(), test_data.end(), n);

  lib::halo_segment<int> segment(2);
  for (std::size_t i = 0; i < n; i++) {
    segment.indices().push_back(i);
  }
  segment.finalize();

  EXPECT_EQ(segment.indices().size(), n);
  for (std::size_t i = 0; i < n; i++) {
    EXPECT_EQ(segment.indices()[i], i);
  }

  segment.pack(test_data.data());
  std::vector<int> unpack_data(10);
  segment.unpack(unpack_data.data());

  expect_eq(comm_rank, test_data, unpack_data);
}

TEST(CpuMpiTests, Halo) {
  const std::size_t n = 10;
  std::vector<int> out(n);
  std::iota(out.begin(), out.end(), comm_rank * 10);

  // Send a halo around a ring
  auto right = (comm_rank + 1) % comm_size;
  lib::halo_segment<int> out_segment(right, {n - 1});
  lib::halo<int> out_halo(comm, {out_segment});

  auto left = (comm_rank + comm_size - 1) % comm_size;
  lib::halo_segment<int> in_segment(left, {0});
  lib::halo<int> in_halo(comm, {in_segment});

  std::vector<int> in(n, 0);

  in_halo.receive();
  // Pass an interator
  out_halo.pack(out.begin());
  out_halo.send();

  in_halo.wait();
  // Pass a pointer
  in_halo.unpack(in.data());

  EXPECT_EQ(in[0], left * n + n - 1);
}
