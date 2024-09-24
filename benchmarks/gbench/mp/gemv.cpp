
#include "mpi.h"

#include "dr/mp.hpp"
#include <fmt/core.h>
#include <sstream>
#include <filesystem>
#include <random>
#include <fstream>

#ifdef STANDALONE_BENCHMARK

MPI_Comm comm;
int comm_rank;
int comm_size;

#else

#include "../common/dr_bench.hpp"

#endif

namespace mp = dr::mp;

#ifdef STANDALONE_BENCHMARK
int main(int argc, char **argv) {

  MPI_Init(&argc, &argv);
  comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &comm_rank);
  MPI_Comm_size(comm, &comm_size);

  if (argc != 3 && argc != 5) {
    fmt::print("usage: ./sparse_benchmark [test outcome dir] [matrix market file], or ./sparse_benchmark [test outcome dir] [number of rows] [number of columns] [number of lower bands] [number of upper bands]\n");
    return 1;
  }
  
#ifdef SYCL_LANGUAGE_VERSION
    sycl::queue q = dr::mp::select_queue();
  mp::init(q);
#else
  mp::init();
#endif
  dr::views::csr_matrix_view<double, long> local_data;
  std::stringstream filenamestream;
  auto root = 0;
  auto computeSize = dr::mp::default_comm().size();
  if (root == dr::mp::default_comm().rank()) {
    if (argc == 5) {
   fmt::print("started loading\n");
        auto n = std::stoul(argv[2]);
        auto up = std::stoul(argv[3]);
        auto down = std::stoul(argv[4]);
        // local_data = dr::generate_random_csr<double, long>({n, m}, density, 42);
        local_data = dr::generate_band_csr<double,long>(n, up, down);
        filenamestream << "mp_band_" << computeSize << "_" << n << "_" << up + down << "_" << local_data.size();
    fmt::print("finished loading\n");
    }
    else {
   fmt::print("started loading\n");
        std::string fname(argv[2]);
        std::filesystem::path p(argv[2]);
        local_data = dr::read_csr<double, long>(fname);
        filenamestream << "mp_" << p.stem().string() << "_" << computeSize << "_" << local_data.size();
    fmt::print("finished loading\n");
    }
  }
  std::string resname;
mp::distributed_sparse_matrix<
    double, long, dr::mp::MpiBackend,
    dr::mp::csr_eq_distribution<double, long, dr::mp::MpiBackend>>
    m_eq(local_data, root);
mp::distributed_sparse_matrix<
    double, long, dr::mp::MpiBackend,
    dr::mp::csr_row_distribution<double, long, dr::mp::MpiBackend>>
    m_row(local_data, root);
  fmt::print("finished distribution\n");
    std::vector<double> eq_duration;
    std::vector<double> row_duration;

    auto N = 10;
    std::vector<double> b;
    b.reserve(m_row.shape().second);
    std::vector<double> res(m_row.shape().first);
    for (auto i = 0; i < m_row.shape().second; i++) {
        b.push_back(i);
    }

    dr::mp::broadcasted_vector<double> allocated_b;
    allocated_b.broadcast_data(m_row.shape().second, 0, b, dr::mp::default_comm());

    fmt::print("started initial gemv distribution\n");
    gemv(0, res, m_eq, allocated_b); // it is here to prepare sycl for work

    fmt::print("finished initial gemv distribution\n");
    for (auto i = 0; i < N; i++) {
      auto begin = std::chrono::high_resolution_clock::now();
      gemv(0, res, m_eq, allocated_b);
      auto end = std::chrono::high_resolution_clock::now();
      double duration = std::chrono::duration<double>(end - begin).count() * 1000;
      eq_duration.push_back(duration);
    }
    
    gemv(0, res, m_row, allocated_b); // it is here to prepare sycl for work
    for (auto i = 0; i < N; i++) {
      auto begin = std::chrono::high_resolution_clock::now();
      gemv(0, res, m_row, allocated_b);
      auto end = std::chrono::high_resolution_clock::now();
      double duration = std::chrono::duration<double>(end - begin).count() * 1000;
      row_duration.push_back(duration);
    }

    if (root == dr::mp::default_comm().rank()) {     
        std::string tmp;
        filenamestream >> tmp;
        std::filesystem::path p(argv[1]);
        p += tmp;
        p += ".csv";
        std::ofstream write_stream(p.string());
        write_stream << eq_duration.front();
        for (auto i = 1; i < N; i++) {
            write_stream << "," << eq_duration[i];
        }
        write_stream << "\n";
        write_stream << row_duration.front();
        for (auto i = 1; i < N; i++) {
            write_stream << "," << row_duration[i];
        }
        write_stream << "\n";
    }
  allocated_b.destroy_data();
  mp::finalize();
}

#else


static void GemvEq_DR(benchmark::State &state) {
  // fft requires usm shared allocation
  std::size_t n = default_vector_size;
  std::size_t up = default_vector_size / 10;
  std::size_t down = default_vector_size / 10;
  assert(dr::mp::use_sycl());
  dr::views::csr_matrix_view<double, long> local_data;
  local_data = dr::generate_band_csr<double,long>(n, up, down);


mp::distributed_sparse_matrix<
    double, long, dr::mp::MpiBackend,
    dr::mp::csr_eq_distribution<double, long, dr::mp::MpiBackend>>
    m(local_data, 0);
   std::vector<double> b;
    b.reserve(m.shape().second);
    std::vector<double> res(m.shape().first);
    for (auto i = 0; i < m.shape().second; i++) {
        b.push_back(i);
    }

    dr::mp::broadcasted_vector<double> allocated_b;
    allocated_b.broadcast_data(m.shape().second, 0, b, dr::mp::default_comm());

  for (auto _ : state) {
    gemv(0, res, m, allocated_b);
  }
}

DR_BENCHMARK(GemvEq_DR);

static void GemvRow_DR(benchmark::State &state) {
  // fft requires usm shared allocation
  std::size_t n = default_vector_size;
  std::size_t up = default_vector_size / 10;
  std::size_t down = default_vector_size / 10;
  assert(dr::mp::use_sycl());
  dr::views::csr_matrix_view<double, long> local_data;
  local_data = dr::generate_band_csr<double,long>(n, up, down);


mp::distributed_sparse_matrix<
    double, long, dr::mp::MpiBackend,
    dr::mp::csr_row_distribution<double, long, dr::mp::MpiBackend>>
    m(local_data, 0);
   std::vector<double> b;
    b.reserve(m.shape().second);
    std::vector<double> res(m.shape().first);
    for (auto i = 0; i < m.shape().second; i++) {
        b.push_back(i);
    }

    dr::mp::broadcasted_vector<double> allocated_b;
    allocated_b.broadcast_data(m.shape().second, 0, b, dr::mp::default_comm());

  for (auto _ : state) {
    gemv(0, res, m, allocated_b);
  }
}

DR_BENCHMARK(GemvRow_DR);



static void Gemv_reference(benchmark::State &state) {
  T actual{};
  std::size_t n = default_vector_size;
  std::size_t up = default_vector_size / 10;
  std::size_t down = default_vector_size / 10;
  assert(dr::mp::use_sycl());
  dr::views::csr_matrix_view<double, long> local_data;
  local_data = dr::generate_band_csr<double,long>(n, up, down);
  auto nnz_count = local_data.size();
  auto band_shape = local_data.shape();
  auto q = get_queue();
  auto policy = oneapi::dpl::execution::make_device_policy(q);
  auto val_ptr = sycl::malloc_device<double>(nnz_count, q);
  auto col_ptr = sycl::malloc_device<long>(nnz_count, q);
  auto row_ptr = sycl::malloc_device<long>(band_shape[0], q);
  std::vector<double> b(band_shape[1]);
  for (auto i = 0; i < band_shape[1]; i++) {
      b.push_back(i);
  }
  auto input = sycl::malloc_device<double>(band_shape[1], q);
  auto output = sycl::malloc_device<double>(band_shape[0], q);

  std::copy(policy, local_src.values_data(), local_src.values_data() + nnz_count, val_ptr);
  std::copy(policy, local_src.colind_data(), local_src.colind_data() + nnz_count, col_ptr);
  std::copy(policy, local_src.rowptr_data(), local_src.rowptr_data() + band_shape[0], row_ptr);
  std::copy(policy, b.begin(), b.end(), input);
  
  for (auto _ : state) {
    q.fill(output, 0, band_shape[0]).wait();
    q.submit([&](auto &cgh) {
            cgh.parallel_for(sycl::range<1>{band_shape[0]}, [=](auto idx) {
              double sum = 0;
              for (auto i = rows_data[idx]; i < rows_data[idx + 1]; i++) {
                auto colNum = local_cols[i];
                auto matrixVal = vals[colNum];
                auto vectorVal = local_vals[i];
                sum += matrixVal * vectorVal;
              }
              *(res + idx) += sum;
            });
          })
          .wait();
  }
  sycl::free(val_ptr, q);
  sycl::free(col_ptr, q);
  sycl::free(row_ptr, q);
  sycl::free(input, q);
  sycl::free(output, q);
}


DR_BENCHMARK(GemvEq_Reference);

DR_BENCHMARK(GemvRow_Reference);

#endif