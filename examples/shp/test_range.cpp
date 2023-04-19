// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <dr/shp/shp.hpp>
#include <sycl/sycl.hpp>
#include <vector>

std::vector<dr::shp::device_ptr<int>> ptrs;

template <typename T>
auto allocate_device_span(std::size_t size, std::size_t rank,
                          sycl::context context, auto &&devices) {
  auto data =
      dr::shp::device_allocator<T>(context, devices[rank]).allocate(size);
  ptrs.push_back(data);

  return dr::shp::device_span<T, decltype(data)>(data, size, rank);
}

template <typename T>
auto allocate_device_spans(std::size_t size, sycl::context context,
                           auto &&devices) {
  std::vector<dr::shp::device_span<T, dr::shp::device_ptr<T>>> spans;
  for (std::size_t rank = 0; rank < devices.size(); rank++) {
    spans.push_back(allocate_device_span<T>(size, rank, context, devices));
  }
  return spans;
}

template <typename T>
dr::shp::device_span<T> allocate_shared_span(std::size_t size, std::size_t rank,
                                             auto &&devices) {
  sycl::queue q(devices[rank]);
  T *data = sycl::malloc_shared<T>(size, q);

  return dr::shp::device_span<T>(data, size, rank);
}

template <typename T>
std::vector<dr::shp::device_span<T>> allocate_shared_spans(std::size_t size,
                                                           auto &&devices) {
  std::vector<dr::shp::device_span<T>> spans;
  for (std::size_t rank = 0; rank < devices.size(); rank++) {
    spans.push_back(allocate_shared_span<T>(size, rank, devices));
  }
  return spans;
}

int main(int argc, char **argv) {
  // Get devices.
  auto devices = dr::shp::get_numa_devices(sycl::default_selector_v);

  dr::shp::init(devices);

  printf("We have %lu devices.\n", devices.size());

  // Total number of elements
  std::size_t size = 200;

  // Elements per rank, ceil(size / devices.size())
  std::size_t size_per_segment = (size + devices.size() - 1) / devices.size();

  auto segments =
      allocate_device_spans<int>(size_per_segment, dr::shp::context(), devices);

  dr::shp::distributed_span dspan(segments);

  printf("Launching on segments...\n");
  std::size_t iteration = 0;
  for (auto &&segment : dspan.segments()) {
    printf("Segment %lu\n", iteration++);
    // sycl::queue q(dr::shp::context(), devices[segment.rank()]);
    sycl::queue q(devices[0]);
    int *ptr = segment.begin().local();
    q.parallel_for(segment.size(), [=](auto id) { ptr[id] = id; }).wait();
  }

  auto subspan = dspan.subspan(25, 70);

  dr::shp::print_range(subspan);

  auto policy = dr::shp::par_unseq;

  auto r_sub = dr::shp::reduce(policy, subspan, 0.0f, std::plus());
  printf("r_sub: %f\n", r_sub);

  dr::shp::print_range(dspan);

  dr::shp::for_each(policy, dspan, [](auto &&elem) { elem = elem + 2; });

  dr::shp::print_range(dspan);

  auto r = dr::shp::reduce(policy, dspan, 0.0f, std::plus());
  printf("r: %f\n", r);

  return 0;
}
