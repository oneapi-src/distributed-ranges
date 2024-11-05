
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

namespace {
  std::size_t getWidth() {
    return 8;//default_vector_size / 100000;
  }
}
static auto getMatrix() {
  std::size_t n = std::sqrt(default_vector_size / 100000) * 50000;
  // std::size_t n = default_vector_size / 2;
  std::size_t up = n / 50;
  std::size_t down = n / 50;
  // assert(dr::mp::use_sycl());
  // assert(dr::mp::sycl_mem_kind() == sycl::usm::alloc::device);
  return dr::generate_band_csr<double,long>(n, up, down);

  // return dr::read_csr<double, long>("/home/komarmik/examples/soc-LiveJournal1.mtx");
  // return dr::read_csr<double, long>("/home/komarmik/examples/mycielskian18.mtx");
  // return dr::read_csr<double, long>("/home/komarmik/examples/mawi_201512020030.mtx");
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
        base_a[i + j * n] = i*j + 1;
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
  // fft requires usm shared allocation
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
        base_a[i + j * n] = i*j + 1;
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
  double* elems = new double[band_shape[0] * width];
  auto input = sycl::malloc_device<double>(band_shape[1] * width, q);
  auto output = sycl::malloc_device<double>(band_shape[0] * width, q);
  //   for (int i = 0; i < band_shape[0]; i++) {
  //   fmt::print("{} {}\n", i, local_data.rowptr_data()[i]);
  // }
  q.memcpy(val_ptr, local_data.values_data(), nnz_count * sizeof(double)).wait();
  q.memcpy(col_ptr, local_data.colind_data(), nnz_count * sizeof(long)).wait();
  q.memcpy(row_ptr, local_data.rowptr_data(), (band_shape[0] + 1) * sizeof(long)).wait();
  q.fill(output, 0, band_shape[0] * width);
  // std::copy(policy, local_data.values_data(), local_data.values_data() + nnz_count, val_ptr);
  // std::copy(policy, local_data.colind_data(), local_data.colind_data() + nnz_count, col_ptr);
  // std::copy(policy, local_data.rowptr_data(), local_data.rowptr_data() + band_shape[0], row_ptr);

  std::copy(policy, b.begin(), b.end(), input);
  // for (int i = 0; i < band_shape[0]; i++) {
  //   fmt::print("{} {}\n", i, local_data.rowptr_data()[i + 1] - local_data.rowptr_data()[i]);
  // }

  auto wg = 32;
  while (width * band_shape[0] * wg > INT_MAX) {
    wg /= 2;
  }
  assert(wg > 0);
  
  for (auto _ : state) {
    if (dr::mp::use_sycl()) {
      dr::mp::sycl_queue().submit([&](auto &&h) { 
        h.parallel_for(sycl::nd_range<1>(width * band_shape[0] * wg, wg), [=](auto item) {
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
      }).wait();
      q.memcpy(elems, output, band_shape[0] * sizeof(double) * width).wait();
    }
    else {
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

static void GemvEq_Reference(benchmark::State &state) {
    Gemv_Reference(state);
}

static void GemvRow_Reference(benchmark::State &state) {
    Gemv_Reference(state);
}

DR_BENCHMARK(GemvEq_Reference);

DR_BENCHMARK(GemvRow_Reference);

#endif