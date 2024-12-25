// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "dr_bench.hpp"

template <class ContainerT> void Stream_Copy(benchmark::State &state) {
  using T = rng::iter_value_t<ContainerT>;
  T init = 0;
  ContainerT a(default_vector_size, init);
  ContainerT b(default_vector_size, init);
  Stats stats(state, sizeof(T) * a.size(), sizeof(T) * b.size());

  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      stats.rep();
      xp::for_each(xp::views::zip(a, b),
                   [](auto &&v) { std::get<1>(v) = std::get<0>(v); });
    }
  }
}

inline int val = 0;

template <class ContainerT> void Stream_Scale(benchmark::State &state) {
  using T = rng::iter_value_t<ContainerT>;
  T scalar = val;
  ContainerT a(default_vector_size, scalar);
  ContainerT b(default_vector_size, scalar);
  Stats stats(state, sizeof(T) * a.size(), sizeof(T) * b.size());

  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      stats.rep();
      xp::for_each(xp::views::zip(a, b), [scalar](auto &&v) {
        std::get<1>(v) = scalar * std::get<0>(v);
      });
    }
  }
}

template <class ContainerT> void Stream_Add(benchmark::State &state) {
  using T = rng::iter_value_t<ContainerT>;
  T scalar = val;
  xp::distributed_vector<T> a(default_vector_size, scalar);
  xp::distributed_vector<T> b(default_vector_size, scalar);
  xp::distributed_vector<T> c(default_vector_size, scalar);
  Stats stats(state, sizeof(T) * (a.size() + b.size()), sizeof(T) * c.size());

  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      stats.rep();
      xp::for_each(xp::views::zip(a, b, c), [](auto &&v) {
        std::get<2>(v) = std::get<0>(v) + std::get<1>(v);
      });
    }
  }
}

template <class ContainerT> void Stream_Triad(benchmark::State &state) {
  using T = rng::iter_value_t<ContainerT>;
  T scalar = val;
  ContainerT a(default_vector_size, scalar);
  ContainerT b(default_vector_size, scalar);
  ContainerT c(default_vector_size, scalar);
  Stats stats(state, sizeof(T) * (a.size() + b.size()), sizeof(T) * c.size());

  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      stats.rep();
      xp::for_each(xp::views::zip(a, b, c), [scalar](auto &&v) {
        std::get<2>(v) = std::get<0>(v) + scalar * std::get<1>(v);
      });
    }
  }
}
