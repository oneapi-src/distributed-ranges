#include <dr/mhp.hpp>
#include <fmt/core.h>

namespace mhp = dr::mhp;

int main(int argc, char **argv) {
  mhp::init();

  fmt::print("Hello, World! Distributed ranges is running on rank {} / {} on "
             "host {}\n",
             mhp::rank(), mhp::nprocs(), mhp::hostname());

  std::size_t n = 1000;

  mhp::distributed_vector<int> v(n);

  auto &&segments = v.segments();

  fmt::print("Created distributed_vector of size {} with {} segments.\n",
             v.size(), segments.size());

  std::size_t segment_id = 0;
  for (auto &&segment : segments) {
    fmt::print("Rank {} owns segment {}, which is size {}\n", segment.rank(),
               segment_id, segment.size());
    ++segment_id;
  }

  mhp::finalize();
  return 0;
}
