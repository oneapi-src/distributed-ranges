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
#ifdef SYCL_LANGUAGE_VERSION
  mp::init(sycl::default_selector_v);
#else
  mp::init();
#endif

  dr::views::csr_matrix_view<double, long> local_data;
  auto root = 0;
  if (root == dr::mp::default_comm().rank()) {
    local_data = dr::read_csr<double, long>(fname);
  }
  {
    fmt::print("started\n");
    mp::distributed_sparse_matrix<double, long, dr::mp::MpiBackend, dr::mp::csr_eq_distribution<double, long, dr::mp::MpiBackend>> m(local_data, root);
    fmt::print("hihihih\n");
    mp::distributed_sparse_matrix<double, long, dr::mp::MpiBackend, dr::mp::csr_row_distribution<double, long, dr::mp::MpiBackend>> m_row(local_data, root);
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
    double total_time = 0;
    auto N = 10;
    gemv(0, res, m, a); // it is here to prepare sycl for work
    for (int i = 0; i < N; i++) {
      auto begin = std::chrono::high_resolution_clock::now();
      gemv(0, res, m, a);
      auto end = std::chrono::high_resolution_clock::now();
      double duration = std::chrono::duration<double>(end - begin).count();
      total_time += duration;
      if (i % 10 == 0 && dr::mp::default_comm().rank() == 0) {
        fmt::print("eq canary {}\n", duration);
      }
    }
    fmt::print("eq gemv time {}\n", total_time * 1000 / N);
    m.fence();
    total_time = 0;
    gemv(0, res_row, m_row, a);
    for (int i = 0; i < N; i++) {
      auto begin = std::chrono::high_resolution_clock::now();
      gemv(0, res_row, m_row, a);
      auto end = std::chrono::high_resolution_clock::now();
      double duration = std::chrono::duration<double>(end - begin).count();
      total_time += duration;
      if (i % 10 == 0 && dr::mp::default_comm().rank() == 0) {
        fmt::print("row canary {}\n", duration);
      }
    }
    fmt::print("row gemv time {}\n", total_time * 1000 / N);
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
  
  if (root == dr::mp::default_comm().rank()) {
    dr::__detail::destroy_csr_matrix_view(local_data, std::allocator<double>{});
  }
  mp::finalize();

  return 0;
}
