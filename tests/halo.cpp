#include "cpu-mpi-tests.hpp"

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

  halo::group send_clockwise(right, {n - 1 - radius});
  halo::group receive_clockwise(left, {0});

  halo::group send_counter_clockwise(left, {radius});
  halo::group receive_counter_clockwise(right, {n - radius});

  // pointer
  halo h(comm, d.data(), {send_clockwise, send_counter_clockwise},
         {receive_clockwise, receive_counter_clockwise});

  h.exchange();
  h.wait();

  EXPECT_EQ(d[0], initial_value(left) + n - 2 * radius - 1);
  EXPECT_EQ(d[n - 1], initial_value(right));
}
