// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

namespace dr::mhp {

template <typename DV> class dv_segment_iterator;

template <typename DV> class dv_segment_reference {
  using iterator = dv_segment_iterator<DV>;

public:
  using value_type = typename DV::value_type;

  dv_segment_reference(const iterator it) : iterator_(it) {}

  operator value_type() const { return iterator_.get(); }
  auto operator=(const value_type &value) const {
    iterator_.put(value);
    return *this;
  }
  auto operator=(const dv_segment_reference &other) const {
    *this = value_type(other);
    return *this;
  }
  auto operator&() const { return iterator_; }

private:
  const iterator iterator_;
}; // dv_segment_reference

template <typename DV> class dv_segment_iterator {
public:
  using value_type = typename DV::value_type;
  using size_type = typename DV::size_type;
  using difference_type = typename DV::difference_type;

  dv_segment_iterator() : DV(nullptr), segment_index_(99), index_(99){};
  dv_segment_iterator(DV *dv, std::size_t segment_index, std::size_t index) {
    assert(index < 1000);
    dv_ = dv;
    segment_index_ = segment_index;
    index_ = index;
  }

  // Comparison
  bool operator==(const dv_segment_iterator &other) const noexcept {
    assert(dv_ != nullptr && dv_ == other.dv_);
    return index_ == other.index_ && dv_ == other.dv_;
    // return segment_index_ == other.segment_index_ && index_ == other.index_
    // && dv_ == other.dv_;
  }
  auto operator<=>(const dv_segment_iterator &other) const noexcept {
    assert(dv_ != nullptr && dv_ == other.dv_);
    return index_ <=> other.index_;
    // return segment_index_ == other.segment_index_ ? index_ <=> other.index_ :
    // segment_index_ <=> other.segment_index_;
  }

  // Only this arithmetic manipulate internal state
  auto &operator+=(difference_type n) {
    assert(n < 1000000);
    assert(dv_ != nullptr);
    assert(n >= 0 || static_cast<difference_type>(index_) >= -n);
    index_ += n;
    return *this;
  }
  auto &operator-=(difference_type n) { return *this += (-n); }
  difference_type operator-(const dv_segment_iterator &other) const noexcept {
    assert(dv_ != nullptr && dv_ == other.dv_);
    assert(index_ >= other.index_);
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
  auto operator*() const {
    assert(dv_ != nullptr);
    return dv_segment_reference<DV>{*this};
  }
  auto operator[](difference_type n) const {
    assert(dv_ != nullptr);
    return *(*this + n);
  }

  void get(value_type *dst, std::size_t size) const {
    assert(dv_ != nullptr);
    assert(segment_index_ * dv_->segment_size_ + index_ < dv_->size_);
    auto segment_offset = index_ + dv_->halo_bounds_.prev;
    dv_->win_.get(dst, size * sizeof(*dst), segment_index_,
                  segment_offset * sizeof(*dst));
  }

  value_type get() const {
    value_type val;
    get(&val, 1);
    return val;
  }

  void put(const value_type *dst, std::size_t size) const {
    assert(dv_ != nullptr);
    assert(segment_index_ * dv_->segment_size_ + index_ < dv_->size_);
    auto segment_offset = index_ + dv_->halo_bounds_.prev;
    dr::drlog.debug("dv put:: ({}:{}:{})\n", segment_index_, segment_offset,
                    size);
    dv_->win_.put(dst, size * sizeof(*dst), segment_index_,
                  segment_offset * sizeof(*dst));
  }

  void put(const value_type &value) const { put(&value, 1); }

  auto rank() const {
    assert(dv_ != nullptr);
    return segment_index_;
  }

  auto local() const {
    assert(dv_ != nullptr);
    const auto my_process_segment_index = dv_->win_.communicator().rank();

    if (my_process_segment_index == segment_index_) {
      auto retptr = dv_->data_ + index_ + dv_->halo_bounds_.prev;
      dr::drlog.debug("read local, segidx:{}, idx:{}, addr:{}, val:{}\n",
                      segment_index_, index_, static_cast<void *>(retptr),
                      *retptr);
      return retptr;
    }

    assert(!dv_->halo_bounds().periodic); // not implemented

    if (my_process_segment_index + 1 == segment_index_) {
      assert(index_ <=
             dv_->halo_bounds().next); // <= instead of < to cover end() case
      auto retptr =
          dv_->data_ + dv_->halo_bounds().prev + index_ + dv_->segment_size_;
      dr::drlog.debug("read next halo, segidx:{}, idx:{}, addr:{}, val:{}\n",
                      segment_index_, index_, static_cast<void *>(retptr),
                      *retptr);
      return retptr;
    }

    if (my_process_segment_index == segment_index_ + 1) {
      assert(dv_->segment_size_ - index_ <= dv_->halo_bounds().prev);
      auto retptr =
          dv_->data_ + dv_->halo_bounds_.prev + index_ - dv_->segment_size_;
      dr::drlog.debug("read prev halo, segidx:{}, idx:{}, addr:{}, val:{}\n",
                      segment_index_, index_, static_cast<void *>(retptr),
                      *retptr);
      return retptr;
    }

    assert(false); // trying to read non-owned memory
  }

  auto segments() const {
    assert(dv_ != nullptr);
    return dr::__detail::drop_segments(dv_->segments(), segment_index_, index_);
  }
  auto &halo() const {
    assert(dv_ != nullptr);
    return dv_->halo();
  }
  auto &halo_bounds() const {
    assert(dv_ != nullptr);
    return dv_->halo_bounds();
  }

private:
  DV *dv_ = nullptr;
  std::size_t segment_index_;
  std::size_t index_;
}; // dv_segment_iterator

template <typename DV> class dv_segment {
private:
  using iterator = dv_segment_iterator<DV>;

public:
  using difference_type = std::ptrdiff_t;
  dv_segment() : dv_(nullptr) { assert(false); }
  dv_segment(DV *dv, std::size_t segment_index, std::size_t size) {
    dv_ = dv;
    segment_index_ = segment_index;
    size_ = size;
    assert(dv_ != nullptr);
  }

  auto size() const {
    assert(dv_ != nullptr);
    assert(size_ < 100000);
    return size_;
  }

  auto begin() const {
    assert(dv_ != nullptr);
    return iterator(dv_, segment_index_, 0);
  }
  auto end() const {
    assert(dv_ != nullptr);
    return begin() + size();
  }

  auto operator[](difference_type n) const { return *(begin() + n); }

private:
  DV *dv_ = nullptr;
  std::size_t segment_index_;
  std::size_t size_;
}; // dv_segment

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
      dr::drlog.debug("getting local from dv::iter offset:{}\n", offset_);
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
  distributed_vector(std::size_t size = 0,
                     dr::halo_bounds hb = dr::halo_bounds()) {
    init(size, hb, Allocator());
  }

  /// Constructor
  distributed_vector(std::size_t size, value_type fill_value,
                     dr::halo_bounds hb = dr::halo_bounds()) {
    init(size, hb, Allocator());
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
  auto &halo_bounds() const { return halo_bounds_; }

  auto segments() const { return rng::views::all(segments_); }

  void print_myself_to_log(std::string desc) const {
    dr::drlog.debug("DV {}, dataAddr:{}", desc,
                    static_cast<void *>(this->data_));
    for (std::size_t idx = 0; idx < data_size_; ++idx)
      dr::drlog.debug(" - idx:{} val:{} addr:{}", idx, this->data_[idx],
                      static_cast<void *>(this->data_ + idx));
    dr::drlog.debug("\n");
  }

private:
  void init(auto size, auto hb, const auto &allocator) {
    allocator_ = allocator;
    size_ = size;
    auto comm_size = default_comm().size(); // dr-style ignore
    segment_size_ =
        std::max({(size + comm_size - 1) / comm_size, hb.prev, hb.next});
    data_size_ = segment_size_ + hb.prev + hb.next;
    if (size_ > 0) {
      data_ = allocator_.allocate(data_size_);
    }
    halo_ = new dr::span_halo<T>(default_comm(), data_, data_size_, hb);
    std::size_t segment_index = 0;
    for (std::size_t i = 0; i < size; i += segment_size_) {
      segments_.emplace_back(this, segment_index++,
                             std::min(segment_size_, size - i));
    }
    halo_bounds_ = hb;
    win_.create(default_comm(), data_, data_size_ * sizeof(T));
    active_wins().insert(win_.mpi_win());
    fence();
  }

  friend dv_segment_iterator<distributed_vector>;

  std::size_t segment_size_ = 0;
  std::size_t data_size_ = 0;
  T *data_ = nullptr;
  dr::span_halo<T> *halo_;

  dr::halo_bounds halo_bounds_;
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

template <class DV>
rng::reference_wrapper<typename DV::value_type>
local_(dr::mhp::dv_segment_reference<DV> dvref) {
  return *((&dvref).local());
}

} // namespace dr::mhp
