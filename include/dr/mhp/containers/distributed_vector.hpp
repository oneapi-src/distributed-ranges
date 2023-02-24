// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

namespace mhp {

template <typename T> class dv_segment_iterator;

template <typename T> class distributed_vector;

template <typename T> class dv_segment_reference {
  using iterator = dv_segment_iterator<T>;

public:
  dv_segment_reference(const iterator it) : iterator_(it) {}

  operator T() const { return iterator_.get(); }
  auto operator=(const T &value) const {
    iterator_.put(value);
    return *this;
  }
  auto operator=(const dv_segment_reference &other) const {
    *this = T(other);
    return *this;
  }
  auto operator&() const { return iterator_; }

private:
  const iterator iterator_;
}; // dv_segment_reference

template <typename T> class dv_segment_iterator {
public:
  using value_type = T;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  dv_segment_iterator() = default;
  dv_segment_iterator(const distributed_vector<T> *dv,
                      std::size_t segment_index, std::size_t index) {
    dv_ = dv;
    segment_index_ = segment_index;
    index_ = index;
  }

  // Comparison
  bool operator==(const dv_segment_iterator &other) const noexcept {
    return index_ == other.index_ && dv_ == other.dv_;
  }
  auto operator<=>(const dv_segment_iterator &other) const noexcept {
    return index_ <=> other.index_;
  }

  // Only these arithmetics manipulate internal state
  auto &operator-=(difference_type n) {
    index_ -= n;
    return *this;
  }
  auto &operator+=(difference_type n) {
    index_ += n;
    return *this;
  }
  difference_type operator-(const dv_segment_iterator &other) const noexcept {
    return index_ - other.index_;
  }

  // prefix
  auto &operator++() {
    *this += 1;
    return *this;
  }
  auto &operator--() {
    *this -= 1;
    return *this;
  }

  // postfix
  auto operator++(int) {
    auto prev = *this;
    *this += 1;
    return prev;
  }
  auto operator--(int) {
    auto prev = *this;
    *this -= 1;
    return prev;
  }

  auto operator+(difference_type n) const {
    auto p = *this;
    p += n;
    return p;
  }
  auto operator-(difference_type n) const {
    auto p = *this;
    p -= n;
    return p;
  }

  // When *this is not first in the expression
  friend auto operator+(difference_type n, const dv_segment_iterator &other) {
    return other + n;
  }

  // dereference
  auto operator*() const { return dv_segment_reference{*this}; }
  auto operator[](difference_type n) const { return *(*this + n); }

  T get() const {
    auto segment_offset = index_ + dv_->halo_bounds_.prev;
    auto value = dv_->win_.template get<T>(segment_index_, segment_offset);
    lib::drlog.debug("get {} =  ({}:{})\n", value, segment_index_,
                     segment_offset);
    return value;
  }

  void put(const T &value) const {
    auto segment_offset = index_ + dv_->halo_bounds_.prev;
    lib::drlog.debug("put ({}:{}) = {}\n", segment_index_, segment_offset,
                     value);
    dv_->win_.put(value, segment_index_, segment_offset);
  }

  auto rank() const { return segment_index_; }
  auto local() const { return dv_->data_ + index_ + dv_->halo_bounds_.prev; }

private:
  const distributed_vector<T> *dv_ = nullptr;
  std::size_t segment_index_;
  std::size_t index_;
}; // dv_segment_iterator

template <typename T> class dv_segment {
private:
  using iterator = dv_segment_iterator<T>;

public:
  using difference_type = std::ptrdiff_t;
  dv_segment() = default;
  dv_segment(const distributed_vector<T> *dv, std::size_t segment_index,
             std::size_t size) {
    dv_ = dv;
    segment_index_ = segment_index;
    size_ = size;
  }

  auto size() const { return size_; }

  auto begin() const { return iterator(dv_, segment_index_, 0); }
  auto end() const { return begin() + size(); }

  auto operator[](difference_type n) const { return *(begin() + n); }

private:
  const distributed_vector<T> *dv_;
  std::size_t segment_index_;
  std::size_t size_;
}; // dv_segment

template <typename T> class dv_segments : public std::span<dv_segment<T>> {
public:
  dv_segments() {}
  dv_segments(distributed_vector<T> *dv)
      : std::span<dv_segment<T>>(dv->segments_) {
    dv_ = dv;
  }

private:
  const distributed_vector<T> *dv_;
}; // dv_segments

template <typename T> class distributed_vector {
public:
  dv_segments<T> segments() const { return dv_segments_; }

  using value_type = T;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using iterator = lib::normal_distributed_iterator<
      decltype(std::declval<distributed_vector>().segments())>;
  using reference = std::iter_reference_t<iterator>;

  // Do not copy
  distributed_vector(const distributed_vector &) = delete;
  distributed_vector &operator=(const distributed_vector &) = delete;

  distributed_vector(std::size_t size, lib::halo_bounds hb = lib::halo_bounds())
      : segment_size_(std::max(
            {(size + default_comm().size() - 1) / default_comm().size(),
             hb.prev, hb.next})),
        data_size_(segment_size_ + hb.prev + hb.next), data_(new T[data_size_]),
        halo_(default_comm(), data_, data_size_, hb) {
    size_ = size;
    std::size_t segment_index = 0;
    for (std::size_t i = 0; i < size; i += segment_size_) {
      segments_.emplace_back(this, segment_index++,
                             std::min(segment_size_, size - i));
    }
    halo_bounds_ = hb;
    win_.create(default_comm(), data_, data_size_ * sizeof(T));
    active_wins().insert(win_.mpi_win());
    dv_segments_ = dv_segments<T>(this);
    fence();
  }

  ~distributed_vector() {
    fence();
    active_wins().erase(win_.mpi_win());
    win_.free();
    delete[] data_;
    data_ = nullptr;
  }

  auto begin() const { return iterator(segments(), 0, 0); }
  auto end() const { return iterator(segments(), segments().size(), 0); }

  auto size() const { return size_; }
  auto operator[](difference_type n) const { return *(begin() + n); }
  auto &halo() { return halo_; }

private:
  friend dv_segment_iterator<T>;
  friend dv_segments<T>;

  std::size_t segment_size_ = 0;
  std::size_t data_size_ = 0;
  T *data_ = nullptr;
  lib::span_halo<T> halo_;

  lib::halo_bounds halo_bounds_;
  std::size_t size_;
  std::vector<dv_segment<T>> segments_;
  dv_segments<T> dv_segments_;
  lib::rma_window win_;
};

} // namespace mhp

// Needed to satisfy rng::viewable_range
template <typename T>
inline constexpr bool rng::enable_borrowed_range<mhp::dv_segments<T>> = true;
