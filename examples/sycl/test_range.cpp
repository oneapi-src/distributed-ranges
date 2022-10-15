#include <CL/sycl.hpp>
#include <ranges>
#include <shp/shp.hpp>
#include <vector>

std::vector<shp::device_ptr<int>> ptrs;

template <typename T>
auto allocate_device_span(std::size_t size, std::size_t rank,
                          cl::sycl::context context, auto &&devices) {
  auto data = shp::device_allocator<T>(context, devices[rank]).allocate(size);
  ptrs.push_back(data);

  return lib::device_span<T, decltype(data)>(data, size, rank);
}

template <typename T>
auto allocate_device_spans(std::size_t size, cl::sycl::context context,
                           auto &&devices) {
  std::vector<lib::device_span<T, shp::device_ptr<T>>> spans;
  for (size_t rank = 0; rank < devices.size(); rank++) {
    spans.push_back(allocate_device_span<T>(size, rank, context, devices));
  }
  return spans;
}

template <typename T>
lib::device_span<T> allocate_shared_span(std::size_t size, std::size_t rank,
                                         auto &&devices) {
  cl::sycl::queue q(devices[rank]);
  T *data = cl::sycl::malloc_shared<T>(size, q);

  return lib::device_span<T>(data, size, rank);
}

template <typename T>
std::vector<lib::device_span<T>> allocate_shared_spans(std::size_t size,
                                                       auto &&devices) {
  std::vector<lib::device_span<T>> spans;
  for (size_t rank = 0; rank < devices.size(); rank++) {
    spans.push_back(allocate_shared_span<T>(size, rank, devices));
  }
  return spans;
}

int main(int argc, char **argv) {
  namespace sycl = cl::sycl;

  // Get GPU devices.
  sycl::gpu_selector g;
  auto devices = shp::get_numa_devices(g);

  sycl::context context(devices);

  // Let's call `device[i]` rank `i`.

  printf("We have %lu devices.\n", devices.size());

  // Total number of elements
  std::size_t size = 200;

  // Elements per rank, ceil(size / devices.size())
  std::size_t size_per_segment = (size + devices.size() - 1) / devices.size();

  auto segments =
      allocate_device_spans<int>(size_per_segment, context, devices);

  lib::distributed_span dspan(segments);

  for (auto &&segment : dspan.segments()) {
    // sycl::queue q(devices[segment.rank()]);
    sycl::queue q(devices[0]);
    int *ptr = segment.begin().local();
    q.parallel_for(sycl::range<1>(segment.size()), [=](auto id) {
       ptr[id] = id;
     }).wait();
  }

  shp::print_range(dspan);

  shp::device_policy policy(devices);

  shp::for_each(policy, dspan, [](auto &&elem) { elem = elem + 2; });

  shp::print_range(dspan);

  auto r = shp::reduce(policy, dspan, 0.0f, std::plus());

  printf("Reduced to the value %lf\n", r);

  return 0;
}
