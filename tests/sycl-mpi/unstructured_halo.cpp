#include "sycl-mpi-tests.hpp"

using halo = lib::unstructured_halo<int, lib::sycl_memory<int>>;
using group = halo::index_map;

const std::size_t n = 10;

int initial_value(int rank) { return 100 * (rank + 1); }

void TestUnstructuredHalo(sycl::usm::alloc memory_kind) {
  sycl::queue queue;

  std::vector<int> d(n);
  std::iota(d.begin(), d.end(), initial_value(comm_rank));

  if (comm_size != 2) {
    return;
  }

  std::vector<group> owned_groups, halo_groups;
  if (comm_rank == 0) {
    owned_groups.push_back(group(1, {2, 2}));
    halo_groups.push_back(group(1, {7, 8, 9}));
  } else {
    owned_groups.push_back(group(0, {1, 3, 5}));
    halo_groups.push_back(group(0, {8, 9}));
  }

  auto memory = lib::sycl_memory<int>(queue, memory_kind);
  halo h(comm, d.data(), owned_groups, halo_groups, memory);

  h.exchange_begin();
  h.exchange_finalize();

  if (comm_rank == 0) {
    std::vector<int> correct = {100, 101, 102, 103, 104,
                                105, 106, 201, 203, 205};
    EXPECT_TRUE(equal(d, correct));
  } else {
    std::vector<int> correct = {200, 201, 202, 203, 204,
                                205, 206, 207, 102, 102};
    EXPECT_TRUE(equal(d, correct));
  }
}

TEST(SyclMpiTests, UnstructuredHaloShared) {
  TestUnstructuredHalo(sycl::usm::alloc::shared);
  TestUnstructuredHalo(sycl::usm::alloc::device);
}
