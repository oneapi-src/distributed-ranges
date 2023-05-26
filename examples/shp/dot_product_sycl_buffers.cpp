// #include <dr/shp.hpp>
// #include <fmt/core.h>
// #include <fmt/ranges.h>
// #include <oneapi/dpl/algorithm>
// #include <oneapi/dpl/async>
// #include <oneapi/dpl/execution>
// #include <oneapi/dpl/numeric>

// #ifdef USE_MKL
// #include <oneapi/mkl.hpp>
// #endif

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <numeric>

// #include <fmt/core.h>

#include <CL/sycl.hpp>

int main(int argc, char **argv) {

  if (argc != 3) {
    return 1;
  }

  std::size_t dev_num = std::atoi(argv[1]);
  std::size_t vec_size = std::atoi(argv[2]);

  if (vec_size == 0 || dev_num == 0) {
    return 1;
  }

  double sycl_median = 0;

  std::vector<sycl::device> devices;
  auto platforms = sycl::platform::get_platforms();
  std::cout << "platforms" << platforms.size();
  for (auto const &this_platform : sycl::platform::get_platforms()) {
    // you can also set the env variable
    for (auto &dev : this_platform.get_devices(sycl::info::device_type::cpu)) {
      std::cout << dev.get_info<sycl::info::device::name>() << std::endl;
      devices.emplace_back(dev);
      devices.emplace_back(dev);
      // devices.emplace_back(dev);
    }
  }

  if (dev_num > devices.size()) {
    std::cout << "There is not enough available devices" << std::endl;
    std::cout << "Available devices= " << devices.size()
              << ", requested devices= " << dev_num;
    return 0;
  }
  std::cout << "dev num = " << devices.size() << std::endl;
  vec_size = vec_size * 1000 * 1000ull;
  using T = float;

  std::vector<T> x(vec_size, 1);
  std::vector<T> y(vec_size, 1);

  // std::vector<T> results(devices.size(), 1);

  std::vector<sycl::queue> queues;
  std::size_t remainder = vec_size % dev_num;

  std::vector<T> part_results(devices.size(), 0);

  // std::vector<sycl::buffer<T>> buffers_x;
  // std::vector<sycl::buffer<T>> buffers_y;

  sycl::buffer buffer_results{part_results};

  // std::vector<sycl::accessor<T>> accessors_x;
  // std::vector<sycl::accessor<T>> accessors_y;

  std::vector<double> durations;
  durations.reserve(dev_num);

  for (std::size_t i = 0; i < devices.size(); i++) {
    auto begin = std::chrono::high_resolution_clock::now();

    queues.emplace_back(
        sycl::queue(devices[i], sycl::property::queue::in_order{}));
    std::size_t chunk_size = vec_size / dev_num;
    std::cout << "chunk size= " << chunk_size << std::endl;
    if (i == devices.size() - 1 && remainder > 0)
      chunk_size = chunk_size + remainder;

    std::cout << "i= " << i << std::endl;
    std::cout << "start= "
              << std::distance(x.begin(), x.begin() + i * chunk_size)
              << std::endl;
    std::cout << "end= "
              << std::distance(x.begin() + i * chunk_size,
                               x.begin() + (i + 1) * chunk_size - 1)
              << std::endl;
    // buffers_x.emplace_back(sycl::buffer{x.begin()+i*chunk_size,
    // x.begin()+(i+1)*chunk_size-1});
    // buffers_y.emplace_back(sycl::buffer{y.begin()+i*chunk_size,
    // y.begin()+(i+1)*chunk_size-1});

    sycl::buffer x_buffer(x.begin() + i * chunk_size,
                          x.begin() + (i + 1) * chunk_size);
    sycl::buffer y_buffer(y.begin() + i * chunk_size,
                          y.begin() + (i + 1) * chunk_size);
    queues[i].submit([&](sycl::handler &h) {
      sycl::accessor x_accessor(x_buffer, h);
      sycl::accessor y_accessor(y_buffer, h);
      sycl::accessor accessor_results(buffer_results, h);
      h.parallel_for(sycl::range{chunk_size}, [=](sycl::id<1> idx) {
        accessor_results[i] += x_accessor[idx] * y_accessor[idx];
      });
    });

    queues[i].wait();

    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end - begin).count();
    durations.push_back(duration);
  }
  std::sort(durations.begin(), durations.end());
  sycl_median = durations[durations.size() / 2] * 1000;

  T final_result = 0.0;
  sycl::host_accessor results(buffer_results);
  for (std::size_t i = 0; i < devices.size(); i++) {
    std::cout << "results= " << results[i] << std::endl;
    final_result += results[i];
  }
  // float final_result = std::reduce(results.begin(), results.end());

  printf("%zu, %f, %f\n", devices.size(), final_result, sycl_median);
}
