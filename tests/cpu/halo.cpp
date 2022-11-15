#include "cpu-tests.hpp"

using halo = lib::halo<int>;

const std::size_t n = 10;
const int radius = 1;

int initial_value(int rank) { return 100 * (rank + 1); }

TEST(CpuMpiTests, Halo) {

  std::vector<int> d(n);
  std::iota(d.begin() + radius, d.end() - radius, initial_value(comm_rank));

  // Send a halo around a ring
  auto right = (comm_rank + 1) % comm_size;
  auto left = (comm_rank + comm_size - 1) % comm_size;

  halo::group owned_clockwise(right, {n - 1 - radius});
  halo::group owned_counter_clockwise(left, {radius});

  halo::group halos_clockwise(left, {0});
  halo::group halos_counter_clockwise(right, {n - radius});

  // pointer
  halo h(comm, d.data(), {owned_clockwise, owned_counter_clockwise},
         {halos_clockwise, halos_counter_clockwise});

  h.exchange();
  h.wait();

  EXPECT_EQ(d[0], initial_value(left) + n - 2 * radius - 1);
  EXPECT_EQ(d[n - 1], initial_value(right));
}

TEST(CpuMpiTests, UnstructuredHalo) {

  std::vector<int> d(n);
  std::iota(d.begin(), d.end(), initial_value(comm_rank));

  if (comm_size != 2) {
    return;
  }

  std::vector<halo::group> owned, halos;
  if (comm_rank == 0) {
    owned.emplace_back(halo::group(1, {2, 2}));
    halos.emplace_back(halo::group(1, {7, 8, 9}));
  } else {
    owned.emplace_back(halo::group(0, {1, 3, 5}));
    halos.emplace_back(halo::group(0, {8, 9}));
  }

  halo h(comm, d.data(), owned, halos);

  h.exchange();
  h.wait();

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

TEST(CpuMpiTests, UnstructuredHaloReduce) {

  std::vector<int> d(n);
  std::iota(d.begin(), d.end(), initial_value(comm_rank));

  if (comm_size != 2) {
    return;
  }

  std::vector<halo::group> sends, receives;
  if (comm_rank == 0) {
    sends.emplace_back(halo::group(1, {2, 2}));
    receives.emplace_back(halo::group(1, {7, 8, 9}));
  } else {
    sends.emplace_back(halo::group(0, {1, 3, 5}));
    receives.emplace_back(halo::group(0, {8, 9}));
  }

  halo h(comm, d.data(), sends, receives);

  h.reduce();
  h.wait();

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

  std::vector<halo::group> sends, receives;
  if (comm_rank == 0) {
    sends.emplace_back(halo::group(1, {2, 2}));
    receives.emplace_back(halo::group(1, {7, 8, 9}));
  } else {
    sends.emplace_back(halo::group(0, {1, 3, 5}));
    receives.emplace_back(halo::group(0, {8, 9}));
  }

  halo h(comm, d.data(), sends, receives);

  h.reduce(halo::reduction_operator::MAX);
  h.wait();

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
