// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "../common/dr_bench.hpp"

#ifdef SYCL_LANGUAGE_VERSION
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/async>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/numeric>
#endif

using T = double;

static const T init_val = 1;

static void Chunk_1DLoop_Serial(benchmark::State &state) {
  auto size = num_rows * num_columns;
  std::vector<T> a(size, init_val), b(size, init_val);
  for (auto _ : state) {
    for (std::size_t i = 0; i < size; i++) {
      b[i] = a[i] * 2.0;
    }
  }
}
DR_BENCHMARK(Chunk_1DLoop_Serial);

static void Chunk_2DLoop_Serial(benchmark::State &state) {
  auto size = num_rows * num_columns;
  std::vector<T> a(size, init_val), b(size, init_val);

  for (auto _ : state) {
    for (std::size_t i = 0; i < num_rows; i++) {
      for (std::size_t j = 0; j < num_columns; j++) {
        b[i * num_columns + j] = a[i * num_columns + j] * 2.0;
      }
    }
  }
}

DR_BENCHMARK(Chunk_2DLoop_Serial);

static void Chunk_2DIndex_Serial(benchmark::State &state) {
  auto size = num_rows * num_columns;
  std::vector<T> a(size, init_val), b(size, init_val);

  auto a_chunks = rng::views::chunk(a, num_columns);
  auto b_chunks = rng::views::chunk(b, num_columns);

  static_assert(rng::random_access_range<decltype(a_chunks)>);
  for (auto _ : state) {
    for (std::size_t i = 0; i < num_rows; i++) {
      for (std::size_t j = 0; j < num_columns; j++) {
        b_chunks[i][j] = a_chunks[i][j];
      }
    }
  }
}

DR_BENCHMARK(Chunk_2DIndex_Serial);

static void Chunk_2DIters_Serial(benchmark::State &state) {
  auto size = num_rows * num_columns;
  std::vector<T> a(size, init_val), b(size, init_val);

  for (auto _ : state) {
    auto b_it = b.begin();
    for (auto &&a_chunk : rng::views::chunk(a, num_columns)) {
      for (auto &ae : a_chunk) {
        *b_it++ = ae;
      }
    }
  }
}

DR_BENCHMARK(Chunk_2DIters_Serial);

static void ChunkFlattened_1DIters_Serial(benchmark::State &state) {
  auto size = num_rows * num_columns;
  std::vector<T> a(size, init_val), b(size, init_val);

  auto a_flat = a | rng::views::chunk(num_columns) | rng::views::join;

  for (auto _ : state) {
    auto b_it = b.begin();
    for (auto &ae : a_flat) {
      *b_it++ = ae;
    }
  }
}

DR_BENCHMARK(ChunkFlattened_1DIters_Serial);

static void ChunkFlattened_ForEach_Serial(benchmark::State &state) {
  auto size = num_rows * num_columns;
  std::vector<T> a(size, init_val), b(size, init_val);

  auto a_flat = a | rng::views::chunk(num_columns) | rng::views::join;

  for (auto _ : state) {
    auto b_it = b.begin();
    rng::for_each(a_flat, [&b_it](auto &ae) { *b_it++ = ae; });
  }
}

DR_BENCHMARK(ChunkFlattened_ForEach_Serial);

static void ChunkTransformFlatten_ForEach_Serial(benchmark::State &state) {
  auto size = num_rows * num_columns;
  std::vector<T> a(size, init_val), b(size, init_val);

  auto slice = [](auto &&chunk) {
    return rng::subrange(chunk.begin() + 1, chunk.end() - 1);
  };

  auto a_flat = a | rng::views::chunk(num_columns) |
                rng::views::take(num_rows - 1) | rng::views::drop(1) |
                rng::views::transform(slice) | rng::views::join;

  for (auto _ : state) {
    auto b_it = b.begin();
    rng::for_each(a_flat, [&b_it](auto &ae) { *b_it++ = ae; });
  }
}

DR_BENCHMARK(ChunkTransformFlatten_ForEach_Serial);
