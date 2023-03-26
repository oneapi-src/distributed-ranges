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
  zip_view(R &&...r) : base_(rng::views::all(std::forward<R>(r))...) {}

  auto begin() const {
    return lib::normal_distributed_iterator<decltype(segments())>(
        segments(), std::size_t(0), 0);
  }

  auto end() const {
    auto segs = segments();
    return lib::normal_distributed_iterator<decltype(segments())>(
        std::move(segs), std::size_t(segs.size()), 0);
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
    return std::apply(zip_segments, base_);
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
    return rng::zip_view(
               (lib::ranges::local(lib::ranges::segments(&refs)[0]))...)
        .begin();
  };
  return std::apply(refs_to_local_zip_iterator, *zi);
}

} // namespace DR_RANGES_NAMESPACE
