#include "cpu-tests.hpp"

using halo = lib::span_halo<int>;
using group = halo::group_type;

const std::size_t n = 10;
const int radius = 1;

int initial_value(int rank) { return 100 * (rank + 1); }

//
// 100 101 102 103 104 105 106 107 108 109
// 200 201 202 203 204 205 206 207 208 209
//
TEST(CpuMpiTests, SpanHalo) {
  auto right = (comm_rank + 1) % comm_size;
  auto left = (comm_rank + comm_size - 1) % comm_size;

  std::vector<int> d(n);
  std::iota(d.begin() + radius, d.end() - radius, initial_value(comm_rank));

  halo h(comm, d.data(), d.size(), radius);

  h.exchange_begin();
  h.exchange_finalize();

  EXPECT_EQ(d[0], initial_value(left) + n - 2 * radius - 1);
  EXPECT_EQ(d[n - 1], initial_value(right));
}
