#include "cpu-mpi-tests.hpp"

TEST(CpuMpiTests, Communicator) { lib::communicator comm; }

TEST(CpuMpiTests, CommunicatorWin) {
  lib::communicator comm;

  const int sz = 20;
  auto p = new int[sz];

  lib::communicator::win win;
  win.create(comm, p, sz * sizeof(*p));
  win.fence();

  win.fence();
  win.free();
  delete[] p;
}
