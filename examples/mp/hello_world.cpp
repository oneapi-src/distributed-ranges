// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <dr/mp.hpp>
#include <fmt/core.h>

namespace mp = dr::mp;

int main(int argc, char **argv) {
#ifdef SYCL_LANGUAGE_VERSION
  mp::init(sycl::default_selector_v);
#else
  mp::init();
#endif

  {

    fmt::print("Hello, World! Distributed ranges is running on rank {} / {} on "
               "host {}\n",
               mp::rank(), mp::nprocs(), mp::hostname());

    std::size_t n = 1000;

    mp::distributed_vector<int> v(n);

    if (mp::rank() == 0) {
      auto &&segments = v.segments();

      fmt::print("Created distributed_vector of size {} with {} segments.\n",
                 v.size(), segments.size());

      std::size_t segment_id = 0;
      for (auto &&segment : segments) {
        fmt::print("Rank {} owns segment {}, which is size {}\n",
                   dr::ranges::rank(segment), segment_id, segment.size());
        ++segment_id;
      }
    }
  }

  mp::finalize();

  return 0;
}
