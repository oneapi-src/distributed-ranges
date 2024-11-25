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

  dr::views::csr_matrix_view<double, long> local_data;
  auto root = 0;
  if (root == dr::mp::default_comm().rank()) {
      local_data = dr::generate_band_csr<double, long>(100, 2, 2);
  }
  {
    mp::distributed_sparse_matrix<
        double, long, dr::mp::MpiBackend,
        dr::mp::csr_row_distribution<double, long, dr::mp::MpiBackend>>
        m_row(local_data, root);
    auto b = m_row.segments()[0].begin().local();
    auto [ind, val] = *b;
    auto [n, ma] = ind;
    fmt::print("some res 2 {} {} {}\n", val, n, ma);

  }

  if (root == dr::mp::default_comm().rank()) {
    dr::__detail::destroy_csr_matrix_view(local_data, std::allocator<double>{});
  }
  mp::finalize();

  return 0;
}
