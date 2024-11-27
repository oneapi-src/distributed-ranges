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
      local_data = dr::generate_band_csr<double, long>(10, 0, 1);
  }
  {
    mp::distributed_sparse_matrix<
        double, long, dr::mp::MpiBackend,
        dr::mp::csr_row_distribution<double, long, dr::mp::MpiBackend>>
        m_row(local_data, root);

    auto mapper = [] (auto elem) { auto [a, b] = elem; auto [c, d] = a; return c;};
    auto z2 = dr::transform_view(m_row, mapper);
    for (auto x: local_segments(m_row)) {
      for (auto z : x)
        {
          auto [a, b] = z;
          auto [c, d] = a;
          fmt::print("some res {} {} {}\n", b, c, d);
        }
    }

    auto q = dr::mp::sycl_queue();
    auto sum1 = sycl::malloc_shared<long>(1, q);
    auto sum2 = sycl::malloc_shared<long>(1, q);
    auto sum3 = sycl::malloc_shared<double>(1, q);
    auto local_iter = local_segments(m_row);
    for (auto x: local_iter) {
      q.submit([=](auto &&h) {
        h.parallel_for(sycl::nd_range<1>(1,1), [=](auto item) {
            for (auto z : x)
              {
                auto [a, b] = z;
                auto [c, d] = a;
                sum1[0] = sum1[0] + b;
                sum2[0] = sum2[0] + c;
                sum3[0] = sum3[0] + d;
              }
          
        });
      }).wait();
      fmt::print("iter vals {} {} {}\n", sum1[0], sum2[0], sum3[0]);
    }

    auto summer = [](auto x, auto y) { return x + y;};
    auto red2 = dr::mp::reduce(z2, 0, summer);
    fmt::print("reduced row {} {}\n", red2, m_row.size());
  }
  
  if (root == dr::mp::default_comm().rank()) {
    dr::__detail::destroy_csr_matrix_view(local_data, std::allocator<double>{});
  }
  mp::finalize();

  return 0;
}
