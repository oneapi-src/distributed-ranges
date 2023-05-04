#include <dr/shp.hpp>

#include <oneapi/dpl/algorithm>

#include <fmt/core.h>
#include <fmt/ranges.h>

namespace shp = dr::shp;

template <rng::forward_range T>
void range(T&&) {}

template <typename T, typename Allocator = dr::shp::device_allocator<T>>
class fast_queue {
public:

  using index_type = std::size_t;

  using allocator_type = Allocator;

  fast_queue(std::size_t capacity, std::size_t rank)
    : rank_(rank), data_(capacity, Allocator(shp::context(), shp::devices()[rank])) {

    auto&& alloc = data_.get_allocator();

    begin_ = 0;
    end_ = 0;
  }

  template <rng::random_access_range R>
  std::pair<sycl::event, bool> push(R&& r) {
    std::size_t s = rng::size(r);

    // Reserve a spot in the queue
    std::size_t spot = end_handle_().fetch_add(s, std::memory_order_relaxed);

    std::size_t begin_c = begin_handle_();

    if (spot - begin_c > data_.size()) {
      end_handle_() -= s;
      return {sycl::event{}, false};
    } else {
      return {shp::copy_async(r.begin(), r.end(), data_.begin() + spot), true};
    }
  }

  auto begin() {
    std::size_t begin_c = begin_handle_();
    return data_.begin() + (begin_c % data_.size());
  }

  auto end() {
    std::size_t end_c = end_handle_();
    return data_.begin() + (end_c & data_.size());
  }

  std::size_t capacity() const {
    return data_.size();
  }

  std::size_t size() const {

  }

private:
  auto begin_handle_() { return std::atomic_ref(begin_); }
  auto begin_handle_() const { return std::atomic_ref(begin_); }

  auto end_handle_() { return std::atomic_ref(end_); }
  auto end_handle_() const { return std::atomic_ref(end_); }

private:
  std::size_t rank_;
  dr::shp::device_vector<T, Allocator> data_;
  index_type begin_ = 0;
  index_type end_ = 0;
};

template <dr::distributed_range R, typename Compare = std::less<>>
void sort(R&& r, Compare comp = Compare()) {
  using T = rng::range_value_t<R>;
  std::vector<sycl::event> events;

  auto&& segments = dr::ranges::segments(r);

  std::size_t n_segments = std::size_t(rng::size(segments));
  std::size_t n_splitters = n_segments - 1;

  T* medians = sycl::malloc_shared<T>(n_segments * n_splitters, shp::devices()[0], shp::context());

  std::size_t segment_id = 0;

  for (auto&& segment : segments) {
    auto&& q = dr::shp::__detail::queue(dr::ranges::rank(segment));
    auto&& local_policy = dr::shp::__detail::dpl_policy(dr::ranges::rank(segment));

    auto&& local_segment = dr::shp::__detail::local(segment);
    dr::__detail::direct_iterator local_begin(rng::begin(local_segment));
    dr::__detail::direct_iterator local_end(rng::end(local_segment));

    sycl::event s = oneapi::dpl::experimental::sort_async(local_policy, local_begin, local_end);

    double step_size = static_cast<double>(rng::size(segment)) / n_segments;

    auto e = q.submit([&](auto&& h) {
      h.depends_on(s);

      h.parallel_for(n_splitters,
                     [=](auto i) {
                       medians[n_splitters*segment_id + i] = local_begin[step_size * (i+1) + 0.5];
                     });
    });

    events.push_back(e);
    ++segment_id;
  }

  dr::shp::__detail::wait(events);

  auto&& local_policy = dr::shp::__detail::dpl_policy(0);
  oneapi::dpl::experimental::sort_async(local_policy, medians, medians + n_segments*n_splitters).wait();

  double step_size = static_cast<double>(n_segments*n_splitters) / n_segments;

  // - Collect median of medians to get final splitters.
  // - Write splitters to [0, n_splitters) in `medians`
  for (std::size_t i = 0; i < n_splitters; i++) {
    medians[i] = medians[std::size_t(step_size * (i+1) + 0.5)];
  }

  segment_id = 0;
  for (auto&& segment : segments) {
    auto&& q = dr::shp::__detail::queue(dr::ranges::rank(segment));
    auto&& local_policy = dr::shp::__detail::dpl_policy(dr::ranges::rank(segment));

    auto&& local_segment = dr::shp::__detail::local(segment);
    dr::__detail::direct_iterator local_begin(rng::begin(local_segment));
    dr::__detail::direct_iterator local_end(rng::end(local_segment));

    fmt::print("Segment {}: {}\n", dr::ranges::rank(segment), local_segment);

    for (std::size_t i = 0; i < n_splitters; i++) {
      auto lower_bound = oneapi::dpl::lower_bound(local_policy, local_begin, local_end, local_begin, local_end, medians[i], comp);

      fmt::print("Sending {} -> {}\n", rng::subrange(local_begin, lower_bound), i);
      local_begin = lower_bound;
    }

    ++segment_id;
  }

  fmt::print("Medians: ");
  for (std::size_t i = 0; i < n_splitters; i++) {
    fmt::print("{} ", medians[i]);
  }
  fmt::print("\n");
}

int main(int argc, char** argv) {
  auto devices = shp::get_numa_devices(sycl::default_selector_v);
  shp::init(devices);

  std::size_t n = 1000;

  shp::distributed_vector<int> v(n);

  for (std::size_t i = 0; i < v.size(); i++) {
    v[i] = lrand48() % 100;
  }

  sort(v);

  return 0;
}