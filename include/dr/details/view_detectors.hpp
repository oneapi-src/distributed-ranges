// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <type_traits>

namespace lib {

template <typename T> struct is_ref_view : std::false_type {};
template <rng::range R>
struct is_ref_view<rng::ref_view<R>> : std::true_type {};

template <typename T> inline constexpr bool is_ref_view_v = is_ref_view<T>{};

template <typename T> struct is_take_view : std::false_type {};
template <typename T>
struct is_take_view<rng::take_view<T>> : std::true_type {};

template <typename T>
inline constexpr bool is_take_view_v = is_take_view<T>::value;

template <typename T> struct is_drop_view : std::false_type {};
template <typename T>
struct is_drop_view<rng::drop_view<T>> : std::true_type {};

template <typename T>
inline constexpr bool is_drop_view_v = is_drop_view<T>::value;

template <typename T> struct is_subrange_view : std::false_type {};
template <typename T>
struct is_subrange_view<rng::subrange<T>> : std::true_type {};

template <typename T>
inline constexpr bool is_subrange_view_v = is_subrange_view<T>::value;

} // namespace lib
