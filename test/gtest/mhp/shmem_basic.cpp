// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause
#include "iostream"
#include <ishmem.h>
#include <sycl/sycl.hpp>

#include "xhp-tests.hpp"

#ifdef STANDALONE_TEST
MPI_Comm comm;
std::size_t comm_rank;
std::size_t comm_size;
cxxopts::ParseResult options;
#endif

#define CHECK_ALLOC(ptr)                                                       \
  if (ptr == nullptr)                                                          \
    fprintf(stderr, "Could not allocate " #ptr "\n");

int shmem_basic_test() {

  int exit_code = 0;
  constexpr int array_size = 10;

  int my_pe = ishmem_my_pe();
  int npes = ishmem_n_pes();

  sycl::queue q;

  std::cout << "Selected device: "
            << q.get_device().get_info<sycl::info::device::name>() << std::endl;
  std::cout << "Selected vendor: "
            << q.get_device().get_info<sycl::info::device::vendor>()
            << std::endl;

  int *source = (int *)ishmem_malloc(array_size * sizeof(int));
  CHECK_ALLOC(source);
  int *target = (int *)ishmem_malloc(array_size * sizeof(int));
  CHECK_ALLOC(target);
  int *errors = sycl::malloc_host<int>(1, q);
  CHECK_ALLOC(errors);

  printf("[%d] source = %p target = %p\n", my_pe, source, target);

  // Initialize source data
  auto e_init = q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<1>{array_size, array_size},
                   [=](sycl::nd_item<1> idx) {
                     std::size_t i = idx.get_global_id()[0];
                     source[i] = (my_pe << 16) + static_cast<int>(i);
                     target[i] = (my_pe << 16) + 0xface;
                   });
  });
  e_init.wait_and_throw();

  ishmem_barrier_all();

  // Perform get operation
  auto e1 = q.submit([&](sycl::handler &h) {
    h.single_task([=]() {
      int my_dev_pe = ishmem_my_pe();
      int my_dev_npes = ishmem_n_pes();

      ishmem_int_get(target, source, array_size, (my_dev_pe + 1) % my_dev_npes);
    });
  });
  e1.wait_and_throw();

  ishmem_barrier_all();
  *errors = 0;
  // Verify data
  auto e_verify = q.submit([&](sycl::handler &h) {
    h.single_task([=]() {
      for (int i = 0; i < array_size; ++i) {
        if (target[i] != (((my_pe + 1) % npes) << 16) + i) {
          *errors = *errors + 1;
        }
      }
    });
  });
  e_verify.wait_and_throw();

  if (*errors > 0) {
    std::cerr << "[ERROR] Validation check(s) failed: " << *errors << std::endl;
    int *hosttarget = sycl::malloc_host<int>(array_size, q);
    CHECK_ALLOC(hosttarget);
    q.memcpy(hosttarget, target, sizeof(int) * array_size).wait_and_throw();
    for (int i = 0; i < array_size; i += 1) {
      if (hosttarget[i] != (((my_pe + 1) % npes) << 16) + i) {
        fprintf(stdout, "[%d] index %d expected 0x%08x got 0x%08x\n", my_pe, i,
                (((my_pe + 1) % npes) << 16) + i, hosttarget[i]);
      }
    }
    sycl::free(hosttarget, q);
    exit_code = 1;
  } else {
    std::cout << "No errors" << std::endl;
  }

  fflush(stdout);
  sycl::free(errors, q);
  ishmem_free(source);
  ishmem_free(target);

  return exit_code;
}

#ifdef STANDALONE_TEST
int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  comm = MPI_COMM_WORLD;
  {
    int a;
    MPI_Comm_rank(comm, &a);
    comm_rank = a;
    MPI_Comm_size(comm, &a);
    comm_size = a;
  }

  cxxopts::Options options_spec(argv[0], "wave equation");
  // clang-format off
  options_spec.add_options()
    ("n", "Grid size", cxxopts::value<std::size_t>()->default_value("128"))
    ("t,benchmark-mode", "Run a fixed number of time steps.", cxxopts::value<bool>()->default_value("false"))
    ("sycl", "Execute on SYCL device")
    ("l,log", "enable logging")
    ("f,fused-kernel", "Use fused kernels.", cxxopts::value<bool>()->default_value("false"))
    ("device-memory", "Use device memory")
    ("h,help", "Print help");
  // clang-format on

  cxxopts::ParseResult options;
  try {
    options = options_spec.parse(argc, argv);
  } catch (const cxxopts::OptionParseException &e) {
    std::cout << options_spec.help() << "\n";
    exit(1);
  }

  std::unique_ptr<std::ofstream> logfile;
  if (options.count("log")) {
    logfile.reset(new std::ofstream(fmt::format("dr.{}.log", comm_rank)));
    dr::drlog.set_file(*logfile);
  }

  if (options.count("sycl")) {
#ifdef SYCL_LANGUAGE_VERSION
    sycl::queue q = dr::mhp::select_queue();
    std::cout << "Run on: "
              << q.get_device().get_info<sycl::info::device::name>() << "\n";
    dr::mhp::init(q, options.count("device-memory") ? sycl::usm::alloc::device
                                                    : sycl::usm::alloc::shared);
#else
    std::cout << "Sycl support requires icpx\n";
    exit(1);
#endif
  } else {
    if (comm_rank == 0) {
      std::cout << "Run on: CPU\n";
    }
    dr::mhp::init();
  }

  auto error = shmem_basic_test();
  dr::mhp::finalize();
  MPI_Finalize();
  return error;
}
#else
TEST(Shmem, Basic) { shmem_basic_test(); }
#endif
