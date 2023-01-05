// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <ranges>
#include <type_traits>

namespace lib {

template <typename T> struct is_owning_view : std::false_type {};
template <std::ranges::range R>
struct is_owning_view<std::ranges::owning_view<R>> : std::true_type {};

template <typename T>
inline constexpr bool is_owning_view_v = is_owning_view<T>{};

template <typename T> struct is_ref_view : std::false_type {};
template <std::ranges::range R>
struct is_ref_view<std::ranges::ref_view<R>> : std::true_type {};

template <typename T> inline constexpr bool is_ref_view_v = is_ref_view<T>{};

template <typename T> struct is_take_view : std::false_type {};

template <typename T>
struct is_take_view<std::ranges::take_view<T>> : std::true_type {};

template <typename T>
inline constexpr bool is_take_view_v = is_take_view<T>::value;

} // namespace lib
