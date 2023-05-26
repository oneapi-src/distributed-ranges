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
  for (auto const &this_platform : sycl::platform::get_platforms()) {
    // you can also set the env variable
    for (auto &dev : this_platform.get_devices(sycl::info::device_type::cpu)) {
      devices.emplace_back(dev);
    }
  }

  if (dev_num < devices.size()) {
    std::cout << "There is not enough available devices" << std::endl;
    std::cout << "Available devices= " << devices.size()
              << ", requested devices= " << dev_num;
    return 0;
  }

  vec_size = vec_size * 1000 * 1000ull;
  using T = float;

  std::vector<T> x(vec_size, 1);
  std::vector<T> y(vec_size, 1);

  std::vector<T> results(devices.size(), 1);

  std::vector<sycl::queue> queues;
  std::size_t remainder = vec_size % dev_num;
  std::cout << devices.size() << std::endl;

  std::vector<double> durations;
  durations.reserve(dev_num);

  for (std::size_t i = 0; i < devices.size(); i++) {
    auto begin = std::chrono::high_resolution_clock::now();

    queues.emplace_back(
        sycl::queue(devices[i], sycl::property::queue::in_order{}));
    std::size_t chunk_size = vec_size / dev_num;

    if (i == devices.size() - 1 && remainder > 0)
      chunk_size = chunk_size + remainder;

    float *x_chunk_dev = sycl::malloc_device<float>(chunk_size, queues[i]);
    float *y_chunk_dev = sycl::malloc_device<float>(chunk_size, queues[i]);
    float *result_chunk_dev = sycl::malloc_device<float>(1, queues[i]);

    // buffers with accessors compare2?
    // use h.depends_on(sycl::event) to wait for the end of exec of the prev
    // one?
    std::size_t start_chunk;
    if (i == devices.size() - 1 && remainder > 0)
      start_chunk = i * (chunk_size - remainder);
    else
      start_chunk = i * chunk_size;

    queues[i].submit([&](sycl::handler &h) {
      h.memcpy(x_chunk_dev, x.data() + start_chunk, sizeof(float) * chunk_size);
    });

    queues[i].submit([&](sycl::handler &h) {
      h.memcpy(y_chunk_dev, y.data() + start_chunk, sizeof(float) * chunk_size);
    });

    queues[i].submit([&](sycl::handler &h) {
      h.parallel_for(sycl::range{chunk_size},
                     sycl::reduction(result_chunk_dev, std::plus<>()),
                     [=](sycl::id<1> idx, auto &result) {
                       result += x_chunk_dev[idx] * y_chunk_dev[idx];
                     });
    });

    queues[i].submit([&](sycl::handler &h) {
      h.memcpy(&results[i], result_chunk_dev, sizeof(float));
    });

    queues[i].wait();

    sycl::free(x_chunk_dev, queues[i]);
    sycl::free(y_chunk_dev, queues[i]);
    sycl::free(result_chunk_dev, queues[i]);

    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end - begin).count();
    durations.push_back(duration);
  }
  std::sort(durations.begin(), durations.end());
  sycl_median = durations[durations.size() / 2] * 1000;

  float final_result = std::reduce(results.begin(), results.end());

  printf("%zu, %f, %f\n", devices.size(), final_result, sycl_median);
}
