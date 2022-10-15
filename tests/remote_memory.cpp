#include "cpu-mpi-tests.hpp"

TEST(CpuMpiTests, RemotePointerRequirements) {
  using RP = lib::remote_pointer<int>;

  static_assert(std::forward_iterator<RP>);
}
