#include <CL/sycl.hpp>
#include <shp/shp.hpp>

#include <ranges>

#include <iostream>

int main(int argc, char **argv) {
  namespace sycl = cl::sycl;

  printf("Creating NUMA devices...\n");
  auto devices = shp::get_numa_devices(sycl::gpu_selector_v);
  shp::init(devices);

  for (auto &device : devices) {
    std::cout << "  Device: " << device.get_info<sycl::info::device::name>()
              << "\n";
  }

  shp::distributed_vector<int> v(100);

  shp::for_each(shp::par_unseq, shp::enumerate(v), [](auto &&tuple) {
    auto &&[idx, value] = tuple;
    value = idx;
  });

  shp::for_each(shp::par_unseq, v, [](auto &&value) { value += 2; });

  size_t sum = shp::reduce(shp::par_unseq, v, int(0), std::plus{});

  shp::print_range(v);

  std::cout << "Sum: " << sum << std::endl;

  return 0;
}
