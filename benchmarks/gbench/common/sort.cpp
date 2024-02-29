// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "../common/dr_bench.hpp"

using T = float;

template <rng::forward_range X> void fill_random(X &&x) {
  for (auto &&value : x) {
    value = drand48() * 100;
  }
}

class DRSortFixture : public benchmark::Fixture {
protected:
  xhp::distributed_vector<T> *a;
  xhp::distributed_vector<T> *vec;
  std::vector<T> local_vec;

public:
  void SetUp(::benchmark::State &) {
    a = new xhp::distributed_vector<T>(default_vector_size);
    vec = new xhp::distributed_vector<T>(default_vector_size);
    local_vec = std::vector<T>(default_vector_size);
    fill_random(local_vec);
    xhp::copy(local_vec, rng::begin(*a));
  }

  void TearDown(::benchmark::State &state) {
    // copy back to check if last sort really sorted
    xhp::copy(*vec, rng::begin(local_vec));
    delete a;
    delete vec;

    if (!rng::is_sorted(local_vec)) {
      state.SkipWithError("mhp sort did not sort the vector");
    }
  }
};

BENCHMARK_DEFINE_F(DRSortFixture, Sort_DR)(benchmark::State &state) {
  Stats stats(state, sizeof(T) * a->size());
  for (auto _ : state) {
    state.PauseTiming();
    xhp::copy(*a, rng::begin(*vec));
    stats.rep();
    state.ResumeTiming();
    xhp::sort(*vec);
  }
}

DR_BENCHMARK_REGISTER_F(DRSortFixture, Sort_DR);

#ifdef SYCL_LANGUAGE_VERSION
class SyclSortFixture : public benchmark::Fixture {
protected:
  std::vector<T> local_vec;

  sycl::queue queue;
  oneapi::dpl::execution::device_policy<> policy;
  T *vec;

public:
  void SetUp(::benchmark::State &) {
    dr::drlog.debug("setting up SyclSortFixture\n");
    // when using mhp's get_queue() long execution is observed in this test
    // (probably due to JIT), now mhp and shp use their own get_queue-s
    queue = get_queue();
    policy = oneapi::dpl::execution::make_device_policy(queue);
    local_vec = std::vector<T>(default_vector_size);
    fill_random(local_vec);
    vec = sycl::malloc_device<T>(default_vector_size, queue);
  }

  void TearDown(::benchmark::State &state) {
    dr::drlog.debug("tearing down SyclSortFixture\n");
    // copy back to check if last sort really sorted
    queue.memcpy(local_vec.data(), vec, default_vector_size * sizeof(T)).wait();
    sycl::free(vec, queue);

    if (!rng::is_sorted(local_vec)) {
      state.SkipWithError("sycl sort did not sort the vector");
    }
  }
};

BENCHMARK_DEFINE_F(SyclSortFixture, Sort_EXP)(benchmark::State &state) {
  Stats stats(state, sizeof(T) * default_vector_size);

  for (auto _ : state) {

    state.PauseTiming();
    queue.memcpy(vec, local_vec.data(), default_vector_size * sizeof(T)).wait();
    stats.rep();
    state.ResumeTiming();

    std::sort(policy, vec, vec + default_vector_size);
  }
}

DR_BENCHMARK_REGISTER_F(SyclSortFixture, Sort_EXP);

BENCHMARK_DEFINE_F(SyclSortFixture, Sort_Reference)(benchmark::State &state) {
  Stats stats(state, sizeof(T) * default_vector_size);

  for (auto _ : state) {
    state.PauseTiming();
    queue.memcpy(vec, local_vec.data(), default_vector_size * sizeof(T)).wait();
    stats.rep();
    state.ResumeTiming();

    std::span<T> d_a(vec, default_vector_size);
    dr::__detail::direct_iterator d_first(d_a.begin());
    dr::__detail::direct_iterator d_last(d_a.end());
    oneapi::dpl::experimental::sort_async(policy, d_first, d_last,
                                          std::less<>{})
        .wait();
  }
}

DR_BENCHMARK_REGISTER_F(SyclSortFixture, Sort_Reference);
#endif

class StdSortFixture : public benchmark::Fixture {
protected:
  std::vector<T> vec_orig;
  std::vector<T> vec;

public:
  void SetUp(::benchmark::State &) {
    vec_orig = std::vector<T>(default_vector_size);
    fill_random(vec_orig);
  }

  void TearDown(::benchmark::State &state) {
    if (!rng::is_sorted(vec)) {
      state.SkipWithError("std sort did not sort the vector");
    }
  }
};

BENCHMARK_DEFINE_F(StdSortFixture, Sort_Std)(benchmark::State &state) {
  Stats stats(state, sizeof(T) * default_vector_size);

  for (auto _ : state) {
    state.PauseTiming();
    vec = vec_orig;
    stats.rep();
    state.ResumeTiming();
    std::sort(vec.begin(), vec.end());
  }
}

DR_BENCHMARK_REGISTER_F(StdSortFixture, Sort_Std);
