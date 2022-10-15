#pragma once

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
      queues.emplace_back(q);
    }

    for (auto &&event : events) {
      event.wait();
    }
  }
}

} // namespace shp
