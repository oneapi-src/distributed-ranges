// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <dr/mp.hpp>
#include <fmt/core.h>

namespace mp = dr::mp;

int main(int argc, char **argv) {

  if (argc != 2) {
    fmt::print("usage: ./sparse_matrix [matrix market file]\n");
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
  // auto n = 50000;
  // std::size_t up = n / 10;
  // std::size_t down = n / 10;
  // local_data = dr::generate_band_csr<double,long>(n, up, down);
  // if (root == dr::mp::default_comm().rank()) {
  local_data = dr::read_csr<double, long>(fname);
  // }
  {
    mp::distributed_sparse_matrix<
        double, long, dr::mp::MpiBackend,
        dr::mp::csr_eq_distribution<double, long, dr::mp::MpiBackend>>
        m(local_data, root);
    mp::distributed_sparse_matrix<
        double, long, dr::mp::MpiBackend,
        dr::mp::csr_row_distribution<double, long, dr::mp::MpiBackend>>
        m_row(local_data, root);
    fmt::print("{}\n", m.size());

    auto width = 6;
    std::vector<double> res(m.shape().first * width);
    std::vector<double> res_row(m.shape().first * width);
    std::vector<double> base_a(m.shape().second * width);
    for (int j = 0; j < width; j++) {
        for (int i = 0; i < m.shape().second; i++) {
        base_a[i + j * m.shape().second] = i*j + 1;
        }
    }
    

    dr::mp::broadcasted_slim_matrix<double> allocated_a;
    allocated_a.broadcast_data(m_row.shape().second, width, 0, base_a, dr::mp::default_comm());
    m.fence();
    double total_time = 0;
    auto N = 1;
    gemv(0, res, m, allocated_a); // it is here to prepare sycl for work
    for (int i = 0; i < 100; i++) {
      auto begin = std::chrono::high_resolution_clock::now();
      gemv(0, res, m, allocated_a);
      auto end = std::chrono::high_resolution_clock::now();
      double duration = std::chrono::duration<double>(end - begin).count();
      total_time += duration;
      if (root == dr::mp::default_comm().rank()) {
        fmt::print("eq canary {}\n\n", duration * 1000);
      }
    }
    // if (root == dr::mp::default_comm().rank()) {
    //   fmt::print("eq gemv time total {}\n", total_time * 1000 / N);
    // }
    m.fence();
    total_time = 0;
    gemv(0, res_row, m_row, allocated_a);
    for (int i = 0; i < N; i++) {
      auto begin = std::chrono::high_resolution_clock::now();
      gemv(0, res_row, m_row, allocated_a);
      auto end = std::chrono::high_resolution_clock::now();
      double duration = std::chrono::duration<double>(end - begin).count();
      total_time += duration;
      if (i % 10 == 0 && dr::mp::default_comm().rank() == 0) {
        fmt::print("row canary {}\n", duration * 1000);
      }
    }

    if (root == dr::mp::default_comm().rank()) {
      fmt::print("row gemv time total {}\n", total_time * 1000 / N);
    }
    m_row.fence();

    std::vector<double> ref(m.shape().first * width);
    auto res_col_len = m.shape().first;
    if (dr::mp::default_comm().rank() == 0) {
      for (auto a : local_data) {
        auto [index, val] = a;
        auto [m, n] = index;
        for (int i = 0; i < width; i++) {
            ref[m + i * res_col_len] += base_a[n + i * res_col_len] * val;
        }
      }
      for (int i = 0; i < m.shape().first * width; i++) {
        if (res[i] != ref[i]) {
          fmt::print("mismatching outcome {} {} {}\n", i, res[i], ref[i]);
        }
      }
      for (int i = 0; i < m.shape().first * width; i++) {
        if (res_row[i] != ref[i]) {
          fmt::print("mismatching outcome row {} {} {}\n", i, res_row[i], ref[i]);
        }
      }
    }
  allocated_a.destroy_data();
  }

  if (root == dr::mp::default_comm().rank()) {
    dr::__detail::destroy_csr_matrix_view(local_data, std::allocator<double>{});
  }
  mp::finalize();

  return 0;
}
