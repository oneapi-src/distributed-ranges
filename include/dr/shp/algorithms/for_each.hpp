#pragma once

#include "../zip_view.hpp"
#include <CL/sycl.hpp>
#include <concepts/concepts.hpp>
#include <shp/algorithms/execution_policy.hpp>
#include <shp/distributed_span.hpp>

namespace shp {

template <typename ExecutionPolicy, lib::distributed_contiguous_range R,
          typename Fn>
void for_each(ExecutionPolicy &&policy, R &&r, Fn &&fn) {

  namespace sycl = cl::sycl;

  static_assert(
      std::is_same_v<std::remove_cvref_t<ExecutionPolicy>, device_policy>);

  if constexpr (std::is_same_v<std::remove_cvref_t<ExecutionPolicy>,
                               device_policy>) {
    auto &&devices = std::forward<ExecutionPolicy>(policy).get_devices();

    std::vector<sycl::queue> queues;
    std::vector<sycl::event> events;

    for (auto &&segment : r.segments()) {
      auto device = devices[segment.rank()];

      sycl::queue q(device);

      auto begin = segment.begin().local();

      auto event = q.parallel_for(sycl::range<1>(segment.size()),
                                  [=](sycl::id<1> idx) { fn(*(begin + idx)); });
      events.emplace_back(event);
      queues.emplace_back(q);
    }

    for (auto &&event : events) {
      event.wait();
    }
  } else {
    assert(false);
  }
}

namespace {

// For now, just pick the first distributed range in the zip view
// as the primary distributed range.
template <std::size_t I, typename... Ts>
constexpr std::size_t get_primary_distributed_range_() {
  if constexpr (lib::distributed_contiguous_range<std::remove_cvref_t<
                    decltype(std::declval<shp::zip_view<Ts...> &>()
                                 .template get_view<I>())>>) {
    return I;
  } else if constexpr (I < sizeof...(Ts)) {
    return get_primary_distributed_range_<I + 1, Ts...>();
  } else {
    return std::numeric_limits<std::size_t>::max();
  }
}

template <std::size_t I, std::size_t P>
decltype(auto) get_view_or_psegment_(std::size_t offset, std::size_t size,
                                     std::size_t segment_id, auto &&view) {
  if constexpr (I == P) {
    return view.template get_view<I>().segments()[segment_id];
  } else {
    return std::ranges::views::counted(
        view.template get_view<I>().begin() + offset, size);
  }
}

template <std::size_t P, std::size_t... Is>
auto get_view_segment_(std::size_t offset, std::size_t size,
                       std::size_t segment_id, auto &&view,
                       std::index_sequence<Is...>) {
  // return shp::zip_view(std::ranges::views::counted(view. template
  // get_view<Is>().begin() + offset, size)...);
  return shp::zip_view(
      get_view_or_psegment_<Is, P>(offset, size, segment_id, view)...);
}

template <typename... Ts> struct is_zip_view : std::false_type {};

template <typename... Ts>
struct is_zip_view<shp::zip_view<Ts...>> : std::true_type {};

template <typename... Ts>
inline constexpr bool is_zip_view_v = is_zip_view<Ts...>{};

} // namespace

// TODO: support multiple distributed ranges
// TODO: move this logic out of for_each and into zip_view
template <typename ExecutionPolicy, typename Fn, typename... Ts>
void for_each(ExecutionPolicy &&policy, shp::zip_view<Ts...> &r, Fn &&fn) {
  namespace sycl = cl::sycl;

  if constexpr (std::is_same_v<std::remove_cvref_t<ExecutionPolicy>,
                               device_policy>) {
    auto &&devices = std::forward<ExecutionPolicy>(policy).get_devices();

    std::vector<sycl::queue> queues;
    std::vector<sycl::event> events;

    constexpr std::size_t prange = get_primary_distributed_range_<0, Ts...>();

    size_t offset = 0;
    size_t segment_id = 0;
    for (auto &&segment : r.template get_view<prange>().segments()) {
      auto device = devices[segment.rank()];

      size_t size = segment.size();

      sycl::queue q(device);

      auto view_segment =
          get_view_segment_<prange>(offset, size, segment_id, r,
                                    std::make_index_sequence<sizeof...(Ts)>{});

      auto begin = view_segment.begin();

      auto event = q.parallel_for(sycl::range<1>(size),
                                  [=](sycl::id<1> idx) { fn(*(begin + idx)); });

      events.emplace_back(event);
      queues.emplace_back(q);
      offset += size;
      segment_id++;
    }

    for (auto &&event : events) {
      event.wait();
    }
  } else {
    assert(false);
  }
}

template <typename ExecutionPolicy, typename Fn, typename... Ts>
void for_each(ExecutionPolicy &&policy, shp::zip_view<Ts...> &&r, Fn &&fn) {
  for_each(policy, r, fn);
}

} // namespace shp
