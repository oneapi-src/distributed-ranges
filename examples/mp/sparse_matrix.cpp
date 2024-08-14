// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <dr/mp.hpp>
#include <dr/sp.hpp>
#include <fmt/core.h>

namespace mp = dr::mp;

int main(int argc, char **argv) {
    
  if (argc != 2) {
    fmt::print("usage: ./gemv_benchmark [matrix market file]\n");
    return 1;
  }

  std::string fname(argv[1]);
  auto local_data = dr::sp::read_csr<float, long>(fname);
#ifdef SYCL_LANGUAGE_VERSION
  mp::init(sycl::default_selector_v);
#else
  mp::init();
#endif

  {
    mp::distributed_sparse_matrix<float, long> m(local_data);
    for (int i = 0; i < dr::mp::default_comm().size(); i++) {
      if (dr::mp::default_comm().rank() == i) {
        auto csr_iter = local_data.begin();
        for (auto [index, val]: m) {
          auto [m, n] = index;
          
          auto [index_csr, val_csr] = *csr_iter;
          auto [m_csr, n_csr] = index;

          assert(m == m_csr && n_csr == n && val == val_csr);
        }
      }
      m.fence();
    }
    dr::sp::__detail::destroy_csr_matrix_view(local_data, std::allocator<float>{});
  }
  mp::finalize();

  return 0;
}
