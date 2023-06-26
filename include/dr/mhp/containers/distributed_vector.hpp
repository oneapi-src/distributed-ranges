// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/mhp/containers/distribution.hpp>
#include <dr/mhp/containers/segment.hpp>

namespace dr::mhp {

/// distributed vector
template <typename T, typename Allocator = dr::mhp::default_allocator<T>>
class distributed_vector {

public:
  using value_type = T;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using allocator_type = Allocator;

  class iterator {
  public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = typename distributed_vector::value_type;
    using difference_type = typename distributed_vector::difference_type;

    iterator() {}
    iterator(const distributed_vector *parent, difference_type offset)
        : parent_(parent), offset_(offset) {}

    auto operator+(difference_type n) const {
      return iterator(parent_, offset_ + n);
    }
    friend auto operator+(difference_type n, const iterator &other) {
      return other + n;
    }
    auto operator-(difference_type n) const {
      return iterator(parent_, offset_ - n);
    }
    auto operator-(iterator other) const { return offset_ - other.offset_; }

    auto &operator+=(difference_type n) {
      offset_ += n;
      return *this;
    }
    auto &operator-=(difference_type n) {
      offset_ -= n;
      return *this;
    }
    auto &operator++() {
      offset_++;
      return *this;
    }
    auto operator++(int) {
      auto old = *this;
      offset_++;
      return old;
    }
    auto &operator--() {
      offset_--;
      return *this;
    }
    auto operator--(int) {
      auto old = *this;
      offset_--;
      return old;
    }

    bool operator==(iterator other) const {
      if (parent_ == nullptr || other.parent_ == nullptr) {
        return false;
      } else {
        return offset_ == other.offset_;
      }
    }
    auto operator<=>(iterator other) const {
      assert(parent_ == other.parent_);
      return offset_ <=> other.offset_;
    }

    auto operator*() const {
      auto segment_size = parent_->segment_size_;
      return parent_
          ->segments()[offset_ / segment_size][offset_ % segment_size];
    }
    auto operator[](difference_type n) const { return *(*this + n); }

    auto local() {
      auto segment_size = parent_->segment_size_;
      return (parent_->segments()[offset_ / segment_size].begin() +
              offset_ % segment_size)
          .local();
    }

    //
    // Support for distributed ranges
    //
    // distributed iterator provides segments
    // remote iterator provides local
    //
    auto segments() {
      return dr::__detail::drop_segments(parent_->segments(), offset_);
    }

  private:
    const distributed_vector *parent_ = nullptr;
    difference_type offset_;
  };

  // Do not copy
  // We need a move constructor for the implementation of reduce algorithm
  distributed_vector(const distributed_vector &) = delete;
  distributed_vector &operator=(const distributed_vector &) = delete;
  distributed_vector(distributed_vector &&) { assert(false); }

  /// Constructor
  distributed_vector(std::size_t size = 0, distribution dist = distribution()) {
    init(size, dist, Allocator());
  }

  /// Constructor
  distributed_vector(std::size_t size, value_type fill_value,
                     distribution dist = distribution()) {
    init(size, dist, Allocator());
    mhp::fill(*this, fill_value);
  }

  ~distributed_vector() {
    fence();
    active_wins().erase(win_.mpi_win());
    win_.free();
    allocator_.deallocate(data_, data_size_);
    data_ = nullptr;
    delete halo_;
  }

  /// Returns iterator to beginning
  auto begin() const { return iterator(this, 0); }
  /// Returns iterator to end
  auto end() const { return begin() + size_; }

  /// Returns size
  auto size() const { return size_; }
  /// Returns reference using index
  auto operator[](difference_type n) const { return *(begin() + n); }
  auto &halo() { return *halo_; }

  auto segments() const { return rng::views::all(segments_); }

private:
  void init(auto size, auto dist, const auto &allocator) {
    allocator_ = allocator;
    size_ = size;
    distribution_ = dist;

    // determine the distribution of data
    auto comm_size = default_comm().size(); // dr-style ignore
    auto hb = dist.halo();
    std::size_t gran = dist.granularity();
    // TODO: make this an error that is reported back to user
    assert(size % gran == 0 && "size must be a multiple of the granularity");
    assert(hb.prev % gran == 0 && "size must be a multiple of the granularity");
    assert(hb.next % gran == 0 && "size must be a multiple of the granularity");
    segment_size_ = gran * std::max({(size / gran + comm_size - 1) / comm_size,
                                     hb.prev / gran, hb.next / gran});

    data_size_ = segment_size_ + hb.prev + hb.next;
    if (size_ > 0) {
      data_ = allocator_.allocate(data_size_);
    }

    halo_ = new span_halo<T>(default_comm(), data_, data_size_, hb);

    std::size_t segment_index = 0;
    for (std::size_t i = 0; i < size; i += segment_size_) {
      segments_.emplace_back(this, segment_index++,
                             std::min(segment_size_, size - i));
    }

    win_.create(default_comm(), data_, data_size_ * sizeof(T));
    active_wins().insert(win_.mpi_win());
    fence();
  }

  friend dv_segment_iterator<distributed_vector>;

  std::size_t segment_size_ = 0;
  std::size_t data_size_ = 0;
  T *data_ = nullptr;
  span_halo<T> *halo_;

  distribution distribution_;
  std::size_t size_;
  std::vector<dv_segment<distributed_vector>> segments_;
  dr::rma_window win_;
  Allocator allocator_;
};

template <typename DR>
concept has_halo_method = dr::distributed_range<DR> && requires(DR &&dr) {
  { rng::begin(dr::ranges::segments(dr)[0]).halo() };
};

auto &halo(has_halo_method auto &&dr) {
  return rng::begin(dr::ranges::segments(dr)[0]).halo();
}

} // namespace dr::mhp
