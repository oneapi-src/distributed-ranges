#include "cpu-tests.hpp"

using halo = lib::unstructured_halo<int>;
using group = halo::group_type;

const std::size_t n = 10;
const int radius = 1;

int initial_value(int rank) { return 100 * (rank + 1); }

//
// 100 101 102 103 104 105 106 107 108 109
// 200 201 202 203 204 205 206 207 208 209
//
TEST(CpuMpiTests, Halo) {

  std::vector<int> d(n);
  std::iota(d.begin() + radius, d.end() - radius, initial_value(comm_rank));

  // Send a halo around a ring
  auto right = (comm_rank + 1) % comm_size;
  auto left = (comm_rank + comm_size - 1) % comm_size;

  group owned_clockwise(d.data(), right, {n - 1 - radius});
  group owned_counter_clockwise(d.data(), left, {radius});

  group halos_clockwise(d.data(), left, {0});
  group halos_counter_clockwise(d.data(), right, {n - radius});

  // pointer
  halo h(comm, {owned_clockwise, owned_counter_clockwise},
         {halos_clockwise, halos_counter_clockwise});

  h.exchange_begin();
  h.exchange_finalize();

  EXPECT_EQ(d[0], initial_value(left) + n - 2 * radius - 1);
  EXPECT_EQ(d[n - 1], initial_value(right));
}

//
// 100 101 102 103 104 105 106 107 108 109
// 200 201 202 203 204 205 206 207 208 209
//
// Rank 0 send 102, 102
// rank 1 receive to
//                                 102 102
//
//
TEST(CpuMpiTests, UnstructuredHalo) {

  std::vector<int> d(n);
  std::iota(d.begin(), d.end(), initial_value(comm_rank));

  if (comm_size != 2) {
    return;
  }

  std::vector<group> owned_groups, halo_groups;
  if (comm_rank == 0) {
    owned_groups.push_back(group(d.data(), 1, {2, 2}));
    halo_groups.push_back(group(d.data(), 1, {7, 8, 9}));
  } else {
    owned_groups.push_back(group(d.data(), 0, {1, 3, 5}));
    halo_groups.push_back(group(d.data(), 0, {8, 9}));
  }

  halo h(comm, owned_groups, halo_groups);

  h.exchange_begin();
  h.exchange_finalize();

  if (comm_rank == 0) {
    std::vector<int> correct = {100, 101, 102, 103, 104,
                                105, 106, 201, 203, 205};
    expect_eq(d, correct);
  } else {
    std::vector<int> correct = {200, 201, 202, 203, 204,
                                205, 206, 207, 102, 102};
    expect_eq(d, correct);
  }
}

//
// 100 101 102 103 104 105 106 107 108 109
// 200 201 202 203 204 205 206 207 208 209
//
// Rank 1 send 208, 209
// rank 0 receive to
//         208
//         209
//
//
TEST(CpuMpiTests, UnstructuredHaloReduce) {

  std::vector<int> d(n);
  std::iota(d.begin(), d.end(), initial_value(comm_rank));

  if (comm_size != 2) {
    return;
  }

  std::vector<group> owned_groups, halo_groups;
  if (comm_rank == 0) {
    owned_groups.push_back(group(d.data(), 1, {2, 2}));
    halo_groups.push_back(group(d.data(), 1, {7, 8, 9}));
  } else {
    owned_groups.push_back(group(d.data(), 0, {1, 3, 5}));
    halo_groups.push_back(group(d.data(), 0, {8, 9}));
  }

  halo h(comm, owned_groups, halo_groups);

  h.reduce_begin();
  h.reduce_finalize(h.plus);

  if (comm_rank == 0) {
    std::vector<int> correct = {100, 101, 519, 103, 104,
                                105, 106, 107, 108, 109};
    expect_eq(d, correct);
  } else {
    std::vector<int> correct = {200, 308, 202, 311, 204,
                                314, 206, 207, 208, 209};
    expect_eq(d, correct);
  }
}

TEST(CpuMpiTests, UnstructuredHaloReduceMax) {

  std::vector<int> d(n);
  std::iota(d.begin(), d.end(), initial_value(comm_rank));

  if (comm_size != 2) {
    return;
  }

  std::vector<group> owned_groups, halo_groups;
  if (comm_rank == 0) {
    owned_groups.push_back(group(d.data(), 1, {2, 2}));
    halo_groups.push_back(group(d.data(), 1, {7, 8, 9}));
  } else {
    owned_groups.push_back(group(d.data(), 0, {1, 3, 5}));
    halo_groups.push_back(group(d.data(), 0, {8, 9}));
  }

  halo h(comm, owned_groups, halo_groups);

  h.reduce_begin();
  h.reduce_finalize(h.max);

  if (comm_rank == 0) {
    std::vector<int> correct = {100, 101, 209, 103, 104,
                                105, 106, 107, 108, 109};
    expect_eq(d, correct);
  } else {
    std::vector<int> correct = {200, 201, 202, 203, 204,
                                205, 206, 207, 208, 209};
    expect_eq(d, correct);
  }
}

TEST(CpuMpiTests, UnstructuredHaloReduceMin) {

  std::vector<int> d(n);
  std::iota(d.begin(), d.end(), initial_value(comm_rank));

  if (comm_size != 2) {
    return;
  }

  std::vector<group> owned_groups, halo_groups;
  if (comm_rank == 0) {
    owned_groups.push_back(group(d.data(), 1, {2, 2}));
    halo_groups.push_back(group(d.data(), 1, {7, 8, 9}));
  } else {
    owned_groups.push_back(group(d.data(), 0, {1, 3, 5}));
    halo_groups.push_back(group(d.data(), 0, {8, 9}));
  }

  halo h(comm, owned_groups, halo_groups);

  h.reduce_begin();
  h.reduce_finalize(h.min);

  if (comm_rank == 0) {
    std::vector<int> correct = {100, 101, 102, 103, 104,
                                105, 106, 107, 108, 109};
    expect_eq(d, correct);
  } else {
    std::vector<int> correct = {200, 107, 202, 108, 204,
                                109, 206, 207, 208, 209};
    expect_eq(d, correct);
  }
}
