// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "mpi.h"

#include "dr/mp.hpp"
#include <filesystem>
#include <fmt/core.h>
#include <fstream>
#include <random>
#include <sstream>
#include "../common/dr_bench.hpp"

namespace mp = dr::mp;

namespace {
std::size_t getWidth() {
  return 8; // default_vector_size / 100000;
}
} // namespace
static auto getMatrix() {
  // size below is useful when testing weak scaling with default vector size
  // using dr-bench it creates matrix which non-zero element count increases
  // linearly when we increase default_vector_size std::size_t n = std::max(1.,
  // std::sqrt(default_vector_size / 100000)) * 50000;

  std::size_t density_scalar = 50;

  std::size_t n =
      std::max(1., std::sqrt(default_vector_size * density_scalar / 2));

  std::size_t up = n / density_scalar;
  std::size_t down = n / density_scalar;
  fmt::print("Generate matrix");
  auto tmp = dr::generate_band_csr<double, long>(n, up, down);
  fmt::print("generated!");
  return tmp;
}

static void GemvEq_DR(benchmark::State &state) {
  auto local_data = getMatrix();

  mp::distributed_sparse_matrix<
      double, long, dr::mp::MpiBackend,
      dr::mp::csr_eq_distribution<double, long, dr::mp::MpiBackend>>
      m(local_data, 0);
  auto n = m.shape()[1];
  auto width = getWidth();
  std::vector<double> base_a(n * width);
  for (int j = 0; j < width; j++) {
    for (int i = 0; i < n; i++) {
      base_a[i + j * n] = i * j + 1;
    }
  }
  dr::mp::broadcasted_slim_matrix<double> allocated_a;
  allocated_a.broadcast_data(n, width, 0, base_a, dr::mp::default_comm());

  std::vector<double> res(m.shape().first * width);
  gemv(0, res, m, allocated_a);
  for (auto _ : state) {
    gemv(0, res, m, allocated_a);
  }
}

DR_BENCHMARK(GemvEq_DR);

static void GemvRow_DR(benchmark::State &state) {
  auto local_data = getMatrix();

  mp::distributed_sparse_matrix<
      double, long, dr::mp::MpiBackend,
      dr::mp::csr_row_distribution<double, long, dr::mp::MpiBackend>>
      m(local_data, 0);
  auto n = m.shape()[1];
  auto width = getWidth();
  std::vector<double> base_a(n * width);
  for (int j = 0; j < width; j++) {
    for (int i = 0; i < n; i++) {
      base_a[i + j * n] = i * j + 1;
    }
  }
  dr::mp::broadcasted_slim_matrix<double> allocated_a;
  allocated_a.broadcast_data(n, width, 0, base_a, dr::mp::default_comm());

  std::vector<double> res(m.shape().first * width);
  gemv(0, res, m, allocated_a);
  for (auto _ : state) {
    gemv(0, res, m, allocated_a);
  }
}

DR_BENCHMARK(GemvRow_DR);

static void Gemv_Reference(benchmark::State &state) {
  auto local_data = getMatrix();
  auto nnz_count = local_data.size();
  auto band_shape = local_data.shape();
  auto q = get_queue();
  auto policy = oneapi::dpl::execution::make_device_policy(q);
  auto val_ptr = sycl::malloc_device<double>(nnz_count, q);
  auto col_ptr = sycl::malloc_device<long>(nnz_count, q);
  auto row_ptr = sycl::malloc_device<long>((band_shape[0] + 1), q);
  std::vector<double> b;
  auto width = getWidth();
  for (auto i = 0; i < band_shape[1] * width; i++) {
    b.push_back(i);
  }
  double *elems = new double[band_shape[0] * width];
  auto input = sycl::malloc_device<double>(band_shape[1] * width, q);
  auto output = sycl::malloc_device<double>(band_shape[0] * width, q);
  q.memcpy(val_ptr, local_data.values_data(), nnz_count * sizeof(double))
      .wait();
  q.memcpy(col_ptr, local_data.colind_data(), nnz_count * sizeof(long)).wait();
  q.memcpy(row_ptr, local_data.rowptr_data(),
           (band_shape[0] + 1) * sizeof(long))
      .wait();
  q.fill(output, 0, band_shape[0] * width);
  std::copy(policy, b.begin(), b.end(), input);

  auto wg = 32;
  while (width * band_shape[0] * wg > INT_MAX) {
    wg /= 2;
  }
  assert(wg > 0);

  for (auto _ : state) {
    if (dr::mp::use_sycl()) {
      dr::mp::sycl_queue()
          .submit([&](auto &&h) {
            h.parallel_for(
                sycl::nd_range<1>(width * band_shape[0] * wg, wg),
                [=](auto item) {
                  auto input_j = item.get_group(0) / band_shape[0];
                  auto idx = item.get_group(0) % band_shape[0];
                  auto local_id = item.get_local_id();
                  auto group_size = item.get_local_range(0);
                  double sum = 0;
                  auto start = row_ptr[idx];
                  auto end = row_ptr[idx + 1];
                  for (auto i = start + local_id; i < end; i += group_size) {
                    auto colNum = col_ptr[i];
                    auto vectorVal = input[colNum + input_j * band_shape[1]];
                    auto matrixVal = val_ptr[i];
                    sum += matrixVal * vectorVal;
                  }
                  sycl::atomic_ref<double, sycl::memory_order::relaxed,
                                   sycl::memory_scope::device>
                      c_ref(output[idx + band_shape[0] * input_j]);
                  c_ref += sum;
                });
          })
          .wait();
      q.memcpy(elems, output, band_shape[0] * sizeof(double) * width).wait();
    } else {
      std::fill(elems, elems + band_shape[0] * width, 0);
      auto local_rows = local_data.rowptr_data();
      auto row_i = 0;
      auto current_row_position = local_rows[1];

      for (int i = 0; i < nnz_count; i++) {
        while (row_i + 1 < band_shape[0] && i >= current_row_position) {
          row_i++;
          current_row_position = local_rows[row_i + 1];
        }
        for (auto j = 0; j < width; j++) {
          auto item_id = row_i + j * band_shape[0];
          auto val_index = local_data.colind_data()[i] + j * band_shape[0];
          auto value = b[val_index];
          auto matrix_value = local_data.values_data()[i];
          elems[item_id] += matrix_value * value;
        }
      }
    }
  }
  delete[] elems;
  sycl::free(val_ptr, q);
  sycl::free(col_ptr, q);
  sycl::free(row_ptr, q);
  sycl::free(input, q);
  sycl::free(output, q);
}

static void GemvEq_Reference(benchmark::State &state) { Gemv_Reference(state); }

static void GemvRow_Reference(benchmark::State &state) {
  Gemv_Reference(state);
}

DR_BENCHMARK(GemvEq_Reference);

DR_BENCHMARK(GemvRow_Reference);
