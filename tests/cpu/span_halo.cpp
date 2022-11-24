#include "cpu-tests.hpp"

using halo = lib::span_halo<int>;
using group = halo::group_type;

const std::size_t n = 10;

int value(int rank, int index) { return (rank + 1) * 100 + index; }

struct stencil_data {
  stencil_data(std::size_t size, int radius, bool periodic) {
    initial.resize(n);
    for (std::size_t i = 0; i < n; i++) {
      initial[i] = value(comm_rank, i);
    }
    ref = test = initial;

    auto prev = (comm_rank - 1 + comm_size) % comm_size;
    ;
    auto next = (comm_rank + 1) % comm_size;

    if (periodic || comm_rank != 0) {
      std::iota(ref.begin(), ref.begin() + radius, value(prev, n - 2 * radius));
    }
    if (periodic || comm_rank != comm_size - 1) {
      std::iota(ref.end() - radius, ref.end(), value(next, radius));
    }
  }

  void check() {
    show();
    if (ref != test) {
      ADD_FAILURE();
    }
  }

  void show(const auto &title, const auto &vec) {
    fmt::print("{:9}", title);
    for (auto d : vec) {
      fmt::print("{:4} ", d);
    }
    fmt::print("\n");
  }

  void show() {
    show("Initial", initial);
    show("Reference", ref);
    show("Test", test);
  }

  std::vector<int> initial, test, ref;
};

TEST(CpuMpiTests, SpanHaloPeriodic) {
  int radius = 2;
  bool periodic = true;
  stencil_data sd(n, radius, periodic);

  halo h(comm, sd.test, radius, periodic);

  h.exchange_begin();
  h.exchange_finalize();

  sd.check();
}

TEST(CpuMpiTests, SpanHaloPeriodicRadius1) {
  int radius = 1;
  bool periodic = true;
  stencil_data sd(n, radius, periodic);

  halo h(comm, sd.test, radius, periodic);

  h.exchange_begin();
  h.exchange_finalize();

  sd.check();
}

TEST(CpuMpiTests, SpanHaloNonPeriodic) {
  int radius = 2;
  bool periodic = false;
  stencil_data sd(n, radius, periodic);

  halo h(comm, sd.test, radius);

  h.exchange_begin();
  h.exchange_finalize();

  sd.check();
}

TEST(CpuMpiTests, SpanHaloPointer) {
  int radius = 2;
  bool periodic = false;
  stencil_data sd(n, radius, periodic);

  halo h(comm, sd.test.data(), sd.test.size(), radius);

  h.exchange_begin();
  h.exchange_finalize();

  sd.check();
}
