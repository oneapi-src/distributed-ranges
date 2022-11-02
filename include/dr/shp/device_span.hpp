#pragma once

#include <shp/span.hpp>
#include <span>

namespace shp {

// A `device_span` is simply a normal `std::span` that's
// been decorated with an extra `rank()` function, showing
// which rank its memory is located on.
// (Thus fulfilling the `remote_range` concept.)
/*
template <class T,
          std::size_t Extent = std::dynamic_extent>
class device_span : public std::span<T, Extent> {
public:
  constexpr device_span() noexcept {}

  template< class It >
  explicit(Extent != std::dynamic_extent)
  constexpr device_span(It first, std::size_t count, std::size_t rank)
    : rank_(rank), std::span<T, Extent>(first, count) {}

  template< class It, class End >
  explicit(Extent != std::dynamic_extent)
  constexpr device_span(It first, End last, std::size_t rank)
    : rank_(rank), std::span<T, Extent>(first, last) {}

  constexpr std::size_t rank() const noexcept {
    return rank_;
  }

private:
  std::size_t rank_;
};
*/

template <typename T, typename Iter = T *>
class device_span : public shp::span<T, Iter> {
public:
  constexpr device_span() noexcept {}

  using value_type = T;
  using size_type = std::size_t;
  using difference_type = std::size_t;
  using reference = std::iter_reference_t<Iter>;

  template <class It>
  constexpr device_span(It first, std::size_t count, std::size_t rank)
      : rank_(rank), shp::span<T, Iter>(first, count) {}

  template <class It, class End>
  constexpr device_span(It first, End last, std::size_t rank)
      : rank_(rank), shp::span<T, Iter>(first, last) {}

  constexpr std::size_t rank() const noexcept { return rank_; }

  device_span first(std::size_t n) const {
    return device_span(this->begin(), this->begin() + n, rank_);
  }

  device_span last(std::size_t n) const {
    return device_span(this->end() - n, this->end(), rank_);
  }

  device_span subspan(std::size_t offset, std::size_t count) const {
    return device_span(this->begin() + offset, this->begin() + offset + count,
                       rank_);
  }

private:
  std::size_t rank_;
};

} // namespace shp
