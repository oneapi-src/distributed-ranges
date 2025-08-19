// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xp-tests.hpp"

#include <dr/concepts/concepts.hpp>
#include <dr/detail/logger.hpp>
#include <dr/detail/onedpl_direct_iterator.hpp>
#include <dr/detail/ranges_shim.hpp>
#include <dr/detail/sycl_utils.hpp>
#include <dr/mp/global.hpp>

template <typename T> class HaloDual : public testing::Test {};

TYPED_TEST_SUITE(HaloDual, ::testing::Types<dr::mp::dual_distributed_vector<int>>);

template <typename DV>
void check_matching(DV &dv, int idx, int expected_value) {
  typename DV::value_type *local_ptr = (dv.begin() + idx).local();
  EXPECT_TRUE(local_ptr != nullptr);
  typename DV::value_type value_on_host;

  if (dr::mp::use_sycl())
    dr::mp::__detail::sycl_copy(local_ptr, &value_on_host);
  else
    value_on_host = *local_ptr;

  DRLOG("checking idx:{} expected:{}", idx, expected_value);
  EXPECT_EQ(value_on_host, expected_value);
}

template <typename DV>
void local_is_accessible_in_halo_region(const int halo_prev,
                                        const int halo_next) {

  DV dv(12, dr::mp::distribution().halo(halo_prev, halo_next));
  DRLOG("local_is_accessible_in_halo_region TEST START, prev:{}, next:{}",
        halo_prev, halo_next);
  iota(dv, 0);
  DRLOG("exchange start");

  dv.halo().exchange();

  // arrays below is function depending on size of communicator-1
  std::array<int, 4> first_local_index____;
  std::array<int, 4> first_nonlocal_index_;
  std::array<int, 4> second_local_index___;
  std::array<int, 4> second_nonlocal_index;
  const int X = 10000; // to mark unused value

  switch (dr::mp::default_comm().rank()) {
  case 0:
    first_local_index____ = {+0, +0, +0, 0};
    first_nonlocal_index_ = {12, +3, +2, 2};
    second_local_index___ = {+X, +9, 10, X};
    second_nonlocal_index = {+X, 12, 12, X};
    break;
  case 1:
    first_local_index____ = {X, 3, +2, 2};
    first_nonlocal_index_ = {X, 9, +4, 4};
    second_local_index___ = {X, X, +8, X};
    second_nonlocal_index = {X, X, 10, X};
    break;
  case 2:
    first_local_index____ = {X, X, 4, +4};
    first_nonlocal_index_ = {X, X, 8, +6};
    second_local_index___ = {X, X, X, 10};
    second_nonlocal_index = {X, X, X, 12};
    break;
  case 3:
    first_local_index____ = {X, X, X, +6};
    first_nonlocal_index_ = {X, X, X, 10};
    second_local_index___ = {X, X, X, +X};
    second_nonlocal_index = {X, X, X, +X};
    break;
  default:
    first_local_index____ = {X, X, X, X};
    first_nonlocal_index_ = {X, X, X, X};
    second_local_index___ = {X, X, X, X};
    second_nonlocal_index = {X, X, X, X};
  }

  const auto c = dr::mp::default_comm().size() - 1;
  auto first_legal_idx = std::max(0, first_local_index____[c] - halo_prev);
  auto first_illegal_idx = std::min(12, first_nonlocal_index_[c] + halo_next);
  auto second_legal_idx = std::max(0, second_local_index___[c] - halo_prev);
  auto second_illegal_idx = std::min(12, second_nonlocal_index[c] + halo_next);

  DRLOG("checking access to idx between first legal {} and first illegal {}, "
        "c:{}",
        first_legal_idx, first_illegal_idx, c);

  for (int idx = first_legal_idx; idx < first_illegal_idx; ++idx) {
    check_matching(dv, idx, idx);
  }

  DRLOG("checking access to idx between second legal {} and second illegal {}, "
    "c:{}",
    second_legal_idx, second_illegal_idx, c);

  for (int idx = second_legal_idx; idx < second_illegal_idx; ++idx) {
    check_matching(dv, idx, idx);
  }

  DRLOG("checks ok");
}

TYPED_TEST(HaloDual, local_is_accessible_in_halo_region_halo_11) {
  local_is_accessible_in_halo_region<TypeParam>(1, 1);
}

TYPED_TEST(HaloDual, local_is_accessible_in_halo_region_halo_10) {
  local_is_accessible_in_halo_region<TypeParam>(1, 0);
}

TYPED_TEST(HaloDual, local_is_accessible_in_halo_region_halo_01) {
  local_is_accessible_in_halo_region<TypeParam>(0, 1);
}

template <typename DV>
void local_is_accessible_in_halo_region__partial(const int halo_prev,
                                                 const int halo_next) {

  DV dv(12, dr::mp::distribution().halo(halo_prev, halo_next));
  DRLOG("local_is_accessible_in_halo_region TEST START, prev:{}, next:{}",
        halo_prev, halo_next);
  iota(dv, 0);
  DRLOG("exchange start");

  dv.halo().exchange();

  // arrays below is function depending on size of communicator-1
  std::array<int, 4> first_segment_begin_;
  std::array<int, 4> first_segment_end___;
  std::array<int, 4> second_segment_begin;
  std::array<int, 4> second_segment_end__;
  const int X = 10000; // to mark unused value

  switch (dr::mp::default_comm().rank()) {
  case 0:
    first_segment_begin_ = {+0, +0, +0, 0};
    first_segment_end___ = {+6, +3, +2, 2};
    second_segment_begin = {+6, +9, 10, X};
    second_segment_end__ = {12, 12, 12, X};
    break;
  case 1:
    first_segment_begin_ = {X, 3, +2, 2};
    first_segment_end___ = {X, 6, +4, 4};
    second_segment_begin = {X, 6, +8, X};
    second_segment_end__ = {X, 9, 10, X};
    break;
  case 2:
    first_segment_begin_ = {X, X, 4, +4};
    first_segment_end___ = {X, X, 6, +6};
    second_segment_begin = {X, X, 6, 10};
    second_segment_end__ = {X, X, 8, 12};
    break;
  case 3:
    first_segment_begin_ = {X, X, X, +6};
    first_segment_end___ = {X, X, X, +8};
    second_segment_begin = {X, X, X, +8};
    second_segment_end__ = {X, X, X, 10};
    break;
  default:
    first_segment_begin_ = {X, X, X, X};
    first_segment_end___ = {X, X, X, X};
    second_segment_begin = {X, X, X, X};
    second_segment_end__ = {X, X, X, X};
  }

  const auto c = dr::mp::default_comm().size() - 1;
  auto first_legal_idx = std::max(0, first_segment_begin_[c] - halo_prev);
  auto first_illegal_idx = std::min(12, first_segment_end___[c] + halo_next);
  auto second_legal_idx = std::max(0, second_segment_begin[c] - halo_prev);
  auto second_illegal_idx = std::min(12, second_segment_end__[c] + halo_next);

  if (first_segment_end___[c] == second_segment_begin[c]) {
    // we own the middle segment
    first_illegal_idx = std::min(12, first_segment_end___[c]);
    second_legal_idx = std::max(0, second_segment_begin[c]);
  }

  constexpr size_t N_STEPS = 5;
  auto foreach_fn = [](auto&& elem) { elem *= 10; };
  int expected_multiplier = 1;

  for (size_t i = 0; i < N_STEPS; i++) {
    expected_multiplier *= 10;

    partial_for_each(dv, foreach_fn);
    dv.halo().partial_exchange();

    for (int idx = first_legal_idx; idx < first_illegal_idx; ++idx) {
      check_matching(dv, idx, idx * expected_multiplier);
    }

    partial_for_each(dv, foreach_fn);
    dv.halo().partial_exchange();

    for (int idx = second_legal_idx; idx < second_illegal_idx; ++idx) {
      check_matching(dv, idx, idx * expected_multiplier);
    }
  }

  DRLOG("checks ok");
}

TYPED_TEST(HaloDual, local_is_accessible_in_halo_region_halo_11__partial) {
  local_is_accessible_in_halo_region__partial<TypeParam>(0, 1);
}

// perf test!

// these are good
// [[maybe_unused]]
// static constexpr size_t DISTRIBUTED_VECTOR_SIZE = 10000000;
// [[maybe_unused]]
// static constexpr size_t HALO_SIZE = 500000;

// these are good
[[maybe_unused]]
static constexpr size_t DISTRIBUTED_VECTOR_SIZE = 1000000;
[[maybe_unused]]
static constexpr size_t HALO_SIZE = 2048;

// [[maybe_unused]]
// static constexpr size_t DISTRIBUTED_VECTOR_SIZE = 100000000;
 
// [[maybe_unused]]
// static constexpr size_t HALO_SIZE = 500000;

[[maybe_unused]]
static constexpr size_t N_STEPS = 100;

[[maybe_unused]]
static constexpr size_t N_KERNEL_STEPS = 2048;

[[maybe_unused]]
static constexpr bool DO_RAMPING_TESTS = true;

[[maybe_unused]]
static constexpr size_t NON_RAMPING_RETRIES = 1;

[[maybe_unused]] 
auto stencil1d_subrange_op = [](auto &center) {
  auto win = &center;
  center = win[-1] + win[0] + win[1];
};

#define KERNEL_WITH_STEPS(NAME, N) \
  [[maybe_unused]] \
  auto NAME = [](auto &center) { \
    auto win = &center;\
    auto result = win[-1] + win[0] + win[1];\
    for (int i = 1; i < N; i++) {\
      if (i % 2 == 0) {\
        result *= i;\
      } else {\
        result /= i;\
      }\
    }\
    center = result;\
    return result;\
  };

KERNEL_WITH_STEPS(stencil1d_subrange_op__heavy, N_KERNEL_STEPS)

KERNEL_WITH_STEPS(kernel_0,  1 << 0)
KERNEL_WITH_STEPS(kernel_1,  1 << 1)
KERNEL_WITH_STEPS(kernel_2,  1 << 2)
KERNEL_WITH_STEPS(kernel_3,  1 << 3)
KERNEL_WITH_STEPS(kernel_4,  1 << 4)
KERNEL_WITH_STEPS(kernel_5,  1 << 5)
KERNEL_WITH_STEPS(kernel_6,  1 << 6)
KERNEL_WITH_STEPS(kernel_7,  1 << 7)
KERNEL_WITH_STEPS(kernel_8,  1 << 8)
KERNEL_WITH_STEPS(kernel_9,  1 << 9)
KERNEL_WITH_STEPS(kernel_10, 1 << 10)
KERNEL_WITH_STEPS(kernel_11, 1 << 11)
KERNEL_WITH_STEPS(kernel_12, 1 << 12)
KERNEL_WITH_STEPS(kernel_13, 1 << 13)
KERNEL_WITH_STEPS(kernel_14, 1 << 14)
KERNEL_WITH_STEPS(kernel_15, 1 << 15)
KERNEL_WITH_STEPS(kernel_16, 1 << 16)
KERNEL_WITH_STEPS(kernel_17, 1 << 17)
KERNEL_WITH_STEPS(kernel_18, 1 << 18)
KERNEL_WITH_STEPS(kernel_19, 1 << 19)
KERNEL_WITH_STEPS(kernel_20, 1 << 20)

[[maybe_unused]]
void perf_test_dual(const size_t size, const size_t halo_size, const size_t steps, const auto& op) {
  dr::mp::dual_distributed_vector<int> dv(size, dr::mp::distribution().halo(halo_size, halo_size));
  DRLOG("perf_test_dual TEST START");
  iota(dv, 0);
  DRLOG("exchange start");

  auto start = std::chrono::high_resolution_clock::now();

  dv.halo().exchange();

  // auto dv_subrange = rng::subrange(dv.begin() + 1, dv.end() - 1);

  for (size_t i = 0; i < steps; i++) {
    dv.halo().partial_exchange_begin();
    partial_for_each(dv, op);
    dv.halo().partial_exchange_finalize();

    dv.halo().partial_exchange_begin();
    partial_for_each(dv, op);
    dv.halo().partial_exchange_finalize();
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "\ttime: " << duration.count() << "us" << std::endl;
}

[[maybe_unused]]
void perf_test_classic(const size_t size, const size_t halo_size, const size_t steps, const auto& op) {
  dr::mp::distributed_vector<int> dv(size, dr::mp::distribution().halo(halo_size, halo_size));
  DRLOG("perf_test TEST START");
  iota(dv, 0);
  DRLOG("exchange start");

  auto start = std::chrono::high_resolution_clock::now();

  dv.halo().exchange();

  // auto dv_subrange = rng::subrange(dv.begin() + 1, dv.end() - 1);

  for (size_t i = 0; i < steps; i++) {
    for_each(dv, op);
    dv.halo().exchange();
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "\ttime: " << duration.count() << "us" << std::endl;
}

// TYPED_TEST(HaloDual, perf_test_dual_dv) {
//   size_t max_size = DISTRIBUTED_VECTOR_SIZE;

//   if (!DO_RAMPING_TESTS) {
//       for (int i = 0; i < NON_RAMPING_RETRIES; i++) {
//         std::cout << "dual size/halo/kernel: " << DISTRIBUTED_VECTOR_SIZE << "/" << HALO_SIZE << "/" << N_KERNEL_STEPS << "\n";
//         perf_test_dual(DISTRIBUTED_VECTOR_SIZE, HALO_SIZE, N_STEPS, stencil1d_subrange_op__heavy);
//       }
//       return;
//   }

//   for (size_t size = 1000; size <= max_size; size *= 10) {
//     for (size_t halo_size = 1; halo_size <= size / 10; halo_size *= 2) {
//       std::cout << "dual size/halo/kernel: " << size << "/" << halo_size << "/" << N_KERNEL_STEPS << "\n";
//       perf_test_dual(size, halo_size, N_STEPS, stencil1d_subrange_op__heavy);
//     }
//   }
// }

// TYPED_TEST(HaloDual, perf_test_classic_dv) {
//   size_t max_size = DISTRIBUTED_VECTOR_SIZE;

//   if (!DO_RAMPING_TESTS) {
//       for (int i = 0; i < NON_RAMPING_RETRIES; i++) {
//         std::cout << "classic size/halo/kernel: " << DISTRIBUTED_VECTOR_SIZE << "/" << HALO_SIZE << "/" << N_KERNEL_STEPS << "\n";
//         perf_test_classic(DISTRIBUTED_VECTOR_SIZE, HALO_SIZE, N_STEPS, stencil1d_subrange_op__heavy);
//       }
//       return;
//   }

//   for (size_t size = 1000; size <= max_size; size *= 10) {
//     for (size_t halo_size = 1; halo_size <= size / 10; halo_size *= 2) {
//       std::cout << "classic size/halo/kernel: " << size << "/" << halo_size << "/" << N_KERNEL_STEPS << "\n";
//       perf_test_classic(size, halo_size, N_STEPS, stencil1d_subrange_op__heavy);
//     }
//   }
// }

#define VARIED_KERNEL_TEST_CASE(vec_size, halo_size, kernel_log_size)\
      std::cout << "dual size/halo/kernel: " << vec_size << "/" << halo_size << "/" << (1 << kernel_log_size) << "\n";\
      perf_test_dual(vec_size, halo_size, N_STEPS, kernel_##kernel_log_size);\
      std::cout << "classic size/halo/kernel: " << vec_size << "/" << halo_size << "/" << (1 << kernel_log_size) << "\n";\
      perf_test_classic(vec_size, halo_size, N_STEPS, kernel_##kernel_log_size);\

TYPED_TEST(HaloDual, perf_test_both) {
  size_t max_size = DISTRIBUTED_VECTOR_SIZE;

  if (!DO_RAMPING_TESTS) {
      for (int i = 0; i < NON_RAMPING_RETRIES; i++) {
        std::cout << "dual size/halo/kernel: " << DISTRIBUTED_VECTOR_SIZE << "/" << HALO_SIZE << "/" << N_KERNEL_STEPS << "\n";
        perf_test_dual(DISTRIBUTED_VECTOR_SIZE, HALO_SIZE, N_STEPS, stencil1d_subrange_op__heavy);
        std::cout << "classic size/halo/kernel: " << DISTRIBUTED_VECTOR_SIZE << "/" << HALO_SIZE << "/" << N_KERNEL_STEPS << "\n";
        perf_test_classic(DISTRIBUTED_VECTOR_SIZE, HALO_SIZE, N_STEPS, stencil1d_subrange_op__heavy);
      }
      return;
  }

  for (size_t size = max_size; size <= max_size; size *= 10) {
    for (size_t halo_size = 1; halo_size <= size / 10; halo_size *= 2) {
      // std::cout << "dual size/halo/kernel: " << size << "/" << halo_size << "/" << N_KERNEL_STEPS << "\n";
      // perf_test_dual(size, halo_size, N_STEPS, stencil1d_subrange_op__heavy);
      // std::cout << "classic size/halo/kernel: " << size << "/" << halo_size << "/" << N_KERNEL_STEPS << "\n";
      // perf_test_classic(size, halo_size, N_STEPS, stencil1d_subrange_op__heavy);
      VARIED_KERNEL_TEST_CASE(size, halo_size, 0)
      VARIED_KERNEL_TEST_CASE(size, halo_size, 1)
      VARIED_KERNEL_TEST_CASE(size, halo_size, 2)
      VARIED_KERNEL_TEST_CASE(size, halo_size, 3)
      VARIED_KERNEL_TEST_CASE(size, halo_size, 4)
      VARIED_KERNEL_TEST_CASE(size, halo_size, 5)
      VARIED_KERNEL_TEST_CASE(size, halo_size, 6)
      VARIED_KERNEL_TEST_CASE(size, halo_size, 7)
      VARIED_KERNEL_TEST_CASE(size, halo_size, 8)
      VARIED_KERNEL_TEST_CASE(size, halo_size, 9)
      VARIED_KERNEL_TEST_CASE(size, halo_size, 10)
      VARIED_KERNEL_TEST_CASE(size, halo_size, 11)
      VARIED_KERNEL_TEST_CASE(size, halo_size, 12)
      VARIED_KERNEL_TEST_CASE(size, halo_size, 13)
      VARIED_KERNEL_TEST_CASE(size, halo_size, 14)
      VARIED_KERNEL_TEST_CASE(size, halo_size, 15)
      VARIED_KERNEL_TEST_CASE(size, halo_size, 16)
      VARIED_KERNEL_TEST_CASE(size, halo_size, 17)
      // VARIED_KERNEL_TEST_CASE(size, halo_size, 18)
      // VARIED_KERNEL_TEST_CASE(size, halo_size, 19)
      // VARIED_KERNEL_TEST_CASE(size, halo_size, 20)
    }
  }
}

// auto is_local = [](const auto &segment) {
//   return dr::ranges::rank(segment) == dr::mp::default_comm().rank();
// };

// auto perf_test_segment_lambda = [](auto &center) { center = center + 1; };

// void perf_test_dual_segment() {
//   dr::mp::dual_distributed_vector<int> dv(DISTRIBUTED_VECTOR_SIZE, dr::mp::distribution().halo(1, 1));

//   auto start = std::chrono::high_resolution_clock::now();
  
//   for (auto &seg : dr::ranges::segments(dv) | rng::views::filter(is_local)) {
//     auto b = dr::ranges::local(rng::begin(seg));
//     auto s = rng::subrange(b, b + rng::distance(seg));

//     for (size_t i = 0; i < N_STEPS; i++) {
//       rng::for_each(s, perf_test_segment_lambda);
//     }
//   }

//   auto end = std::chrono::high_resolution_clock::now();
//   auto duration = duration_cast<std::chrono::microseconds>(end - start);
//   std::cout << "perf_test_dual_segment results: \n\ttime: " << duration.count() << "us" << std::endl;
// }

// void perf_test_classic_segment() {
//   dr::mp::distributed_vector<int> dv(DISTRIBUTED_VECTOR_SIZE, dr::mp::distribution().halo(1, 1));

//   auto start = std::chrono::high_resolution_clock::now();

//   for (auto &seg : dr::ranges::segments(dv) | rng::views::filter(is_local)) {
//     auto b = dr::ranges::local(rng::begin(seg));
//     auto s = rng::subrange(b, b + rng::distance(seg));

//     for (size_t i = 0; i < N_STEPS; i++) {
//       rng::for_each(s, perf_test_segment_lambda);
//     }
//   }

//   auto end = std::chrono::high_resolution_clock::now();
//   auto duration = duration_cast<std::chrono::microseconds>(end - start);
//   std::cout << "perf_test_classic_segment results: \n\ttime: " << duration.count() << "us" << std::endl;
// }

// TYPED_TEST(HaloDual, perf_test_classic_dv_segment) {
//   perf_test_classic_segment();
// }

// TYPED_TEST(HaloDual, perf_test_dual_dv_segment) {
//   perf_test_dual_segment();
// }
