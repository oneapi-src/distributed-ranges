#include <CL/sycl.hpp>
#include <shp/shp.hpp>

#include <ranges>

int main(int argc, char **argv) {
  namespace sycl = cl::sycl;

  auto devices = shp::get_numa_devices(sycl::gpu_selector_v);
  shp::init(devices);

  shp::distributed_vector<int> v(100);

  shp::for_each(shp::par_unseq, shp::enumerate(v), [](auto &&tuple) {
    auto &&[idx, value] = tuple;
    value = idx;
  });

  shp::for_each(shp::par_unseq, v, [](auto &&value) { value += 2; });

  shp::print_range(v);

  return 0;
}
