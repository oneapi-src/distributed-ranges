// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// TODO: use libstdc++ 13.0 or greater if available.

// #define DR_USE_STD_RANGES

#ifdef DR_USE_STD_RANGES

#include <ranges>

namespace rng = ::std::ranges;

#define DR_RANGES_NAMESPACE std::ranges

namespace dr {

template <typename T>
concept input_iterator = std::input_iterator<T>; // dr-style ignore

template <typename I, typename T>
concept output_iterator = std::output_iterator<I, T>; // dr-style ignore

template <typename T>
concept forward_iterator = std::forward_iterator<T>; // dr-style ignore

template <typename T>
concept random_access_iterator =
    std::random_access_iterator<T>; // dr-style ignore

} // namespace dr

#else

#include <range/v3/all.hpp>

namespace rng = ::ranges;

#define DR_RANGES_NAMESPACE ranges

namespace dr {

template <typename T>
concept input_iterator = rng::input_iterator<T>;

template <typename I, typename T>
concept output_iterator = rng::output_iterator<I, T>;

template <typename T>
concept forward_iterator = rng::forward_iterator<T>;

template <typename T>
concept random_access_iterator = rng::random_access_iterator<T>;

} // namespace dr

#endif
