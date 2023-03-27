// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

namespace mhp {

template <typename I>
concept is_zip_iterator =
    std::forward_iterator<I> && requires(I &iter) { std::get<0>(*iter); };

template <rng::viewable_range... R>
class zip_view : public rng::view_interface<zip_view<R...>> {
public:
  zip_view(const zip_view &z) : base_(z.base_) {}
  zip_view(zip_view &&z) : base_(std::move(z.base_)) {}
  template <typename... V>
  zip_view(V &&...v) : base_(rng::views::all(std::forward<V>(v))...) {}

  auto begin() const {
    return lib::normal_distributed_iterator<decltype(segments())>(
        segments(), std::size_t(0), 0);
  }

  auto end() const {
    auto segs = segments();
    return lib::normal_distributed_iterator<decltype(segments())>(
        std::move(segs), std::size_t(rng::distance(segs)), 0);
  }

  auto segments() const {
    auto zip_segments = [](auto &&...base) {
      auto zip_segment = [](auto &&v) {
        auto zip = [](auto &&...refs) { return rng::views::zip(refs...); };
        return std::apply(zip, v);
      };
      return rng::views::zip(lib::ranges::segments(base)...) |
             rng::views::transform(zip_segment);
    };

    auto z = std::apply(zip_segments, base_);
    auto check_aligned = [z](auto &&...base) {
      if (aligned(rng::begin(base)...)) {
        return z;
      } else {
        // return empty on unaligned
        return decltype(z){};
      }
    };

    return std::apply(check_aligned, base_);
  }

  auto base() const { return base_; }

private:
  std::tuple<R...> base_;
};

template <rng::viewable_range... R>
zip_view(R &&...r) -> zip_view<rng::views::all_t<R>...>;

namespace views {

template <rng::viewable_range... R> auto zip(R &&...r) {
  return zip_view(std::forward<R>(r)...);
}

} // namespace views

} // namespace mhp

namespace DR_RANGES_NAMESPACE {

template <mhp::is_zip_iterator ZI> auto local_(ZI zi) {
  auto refs_to_local_zip_iterator = [](auto &&...refs) {
    // Convert the first segment of each component to local and then
    // zip them together, returning the begin() of the zip view
    return rng::begin(rng::zip_view(
        (lib::ranges::local(lib::ranges::segments(&refs)[0]))...));
  };
  return std::apply(refs_to_local_zip_iterator, *zi);
}

} // namespace DR_RANGES_NAMESPACE
