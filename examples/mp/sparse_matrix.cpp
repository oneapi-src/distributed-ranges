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
  auto local_data = dr::read_csr<double, long>(fname);
#ifdef SYCL_LANGUAGE_VERSION
  mp::init(sycl::default_selector_v);
#else
  mp::init();
#endif

  {
    mp::distributed_sparse_matrix<double, long, dr::mp::MpiBackend, dr::mp::csr_eq_distribution<double, long, dr::mp::MpiBackend>> m(local_data);
    mp::distributed_sparse_matrix<double, long, dr::mp::MpiBackend, dr::mp::csr_row_distribution<double, long, dr::mp::MpiBackend>> m_row(local_data);
    fmt::print("{}\n", m.size());
    // for (int i = 0; i < dr::mp::default_comm().size(); i++) {
    //   if (dr::mp::default_comm().rank() == i) {
    //     auto csr_iter = local_data.begin();
    //     int j = 0;
    //     // fmt::print("{}\n", i);
    //     for (auto [index, val]: m) {
    //       auto [m, n] = index;
          
    //       auto [index_csr, val_csr] = *csr_iter;
    //       auto [m_csr, n_csr] = index_csr;
    //       auto check = m == m_csr && n_csr == n && val == val_csr;
    //       if (!check) {
    //         fmt::print("{} {} {} {} {} {} {}\n", j, m, m_csr, n, n_csr, val, val_csr);
    //       }
    //       // assert(check);
    //       csr_iter++;
    //       j++;
    //     }
    //   }
    //   m.fence();
    // }

    std::vector<double> res(m.shape().first);
    std::vector<double> res_row(m.shape().first);
    std::vector<double> a(m.shape().second);
    for (int i = 0; i < a.size(); i++) {
      a[i] = i;
    }
    m.fence();
    gemv(0, res, m, a);
    m.fence();
    gemv(0, res_row, m_row, a);
    m_row.fence();

    std::vector<double> ref(m.shape().first);
    if (dr::mp::default_comm().rank() == 0) {
       for (auto a : local_data) {
          auto [index, val] = a;
          auto [m, n] = index;
          ref[m] += n * val;
       }
       for (int i = 0; i < m.shape().first; i++) {
          if (res[i] != ref[i]) {
            fmt::print("mismatching outcome {} {}\n", res[i], ref[i]);
          }
       }
       for (int i = 0; i < m.shape().first; i++) {
          if (res_row[i] != ref[i]) {
            fmt::print("mismatching outcome row {} {}\n", res_row[i], ref[i]);
          }
       }
    }
  }
  dr::__detail::destroy_csr_matrix_view(local_data, std::allocator<double>{});
  mp::finalize();

  return 0;
}
