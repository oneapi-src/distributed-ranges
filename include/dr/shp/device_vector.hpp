// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/shp/allocators.hpp>
#include <dr/shp/vector.hpp>

namespace shp {

template <typename T, typename Allocator>
class device_vector { // : public shp::vector<T, Allocator> {
public:
  //constexpr device_vector() noexcept {}

  using base = shp::vector<T, Allocator>;

  using value_type = T;
  using size_type = std::size_t;
  using difference_type = std::size_t;
  using allocator_type = Allocator;

  constexpr device_vector(size_type count, const Allocator &alloc,
                          size_type rank)
      : allocator_(alloc), rank_(rank), size_(count), data_(allocator_.allocate(count))
  {
  }

  ~device_vector() noexcept {
    allocator_.deallocate(data_, size_);
  }

  constexpr std::size_t rank() const noexcept { return rank_; }
  size_type size() const noexcept { return size_; }

  auto data() noexcept { return data_; }
  auto data() const noexcept { return data_; }

  auto begin() noexcept { return data_; }
  auto end() noexcept { return begin() + size_; }

  auto begin() const noexcept { return data_; }
  auto end() const noexcept { return begin() + size_; }

  auto operator[](size_type pos) { return *(begin() + pos); }
  auto operator[](size_type pos) const { return *(begin() + pos); }


private:
  allocator_type allocator_;
  std::size_t rank_ = 0;
  size_type size_ = 0;
  typename std::allocator_traits<Allocator>::pointer data_ = nullptr;
};

template <class Alloc>
device_vector(std::size_t, const Alloc, std::size_t)
    -> device_vector<typename Alloc::value_type, Alloc>;

} // namespace shp
