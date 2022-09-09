template <typename T> class distributed_span {
public:
  using local_span_type = remote_span<T>;

  using element_type = T;
  using value_type = std::remove_cv_t<T>;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  using pointer = typename local_span_type::pointer;
  using const_pointer = typename local_span_type::const_pointer;

  using reference = typename local_span_type::reference;
  using const_reference = typename local_span_type::const_reference;

  using joined_view_type = std::ranges::join_view<
      std::ranges::ref_view<std::vector<local_span_type>>>;

  using iterator = distributed_span_iterator<T>;
  using const_iterator = typename iterator::const_iterator;

  constexpr distributed_span() noexcept = default;
  constexpr distributed_span(const distributed_span &) noexcept = default;
  constexpr distributed_span &
  operator=(const distributed_span &) noexcept = default;

  /// Create a distributed span out of a range of remote_span objects
  template <std::ranges::input_range R>
  requires(
      std::is_same_v<std::ranges::range_value_t<R>,
                     local_span_type>) constexpr distributed_span(R &&spans);

  /// Number of elements
  constexpr size_type size() const noexcept;

  /// Size of the span in bytes
  constexpr size_type size_bytes() const noexcept;

  /// Access individual elements
  constexpr reference operator[](size_type idx) const;

  constexpr std::size_t local_size(size_type rank = 0) const noexcept {
    return spans_[rank].size();
  }

  [[nodiscard]] constexpr bool empty() const noexcept { return size() == 0; }

  /// Retrieve a view of the subspans that comprise the distributed_span
  /* view of remote_spans */ get_subspans() const noexcept;

  constexpr distributed_span<element_type>
  subspan(size_type Offset, size_type Count = std::dynamic_extent) const {
    std::vector<local_span_type> new_spans;

    size_type local_id = Offset / local_size();
    size_type local_offset = Offset % local_size();
    size_type local_count = std::min(local_size() - local_offset, Count);

    new_spans.push_back(spans_[local_id].subspan(local_offset, local_count));

    local_id++;

    for (; local_id * local_size() < Offset + Count && local_id < spans_.size();
         local_id++) {
      size_type local_count =
          std::min(local_size(), (Offset + Count) - local_id * local_size());
      new_spans.push_back(spans_[local_id].subspan(0, local_count));
    }

    return distributed_span<element_type>(new_spans);
  }

  constexpr distributed_span<element_type> first(size_type Count) const {
    return subspan(0, Count);
  }

  constexpr distributed_span<element_type> last(size_type Count) const {
    return subspan(size() - Count, Count);
  }

  iterator begin() const { return iterator(spans_, 0); }

  iterator end() const { return iterator(spans_, size()); }

  /*
    iterator local_begin() const {
      return local_joined_view_.begin();
    }
    iterator local_end() const {
      return local_joined_view_.end();
    }
    */

  constexpr reference front() const { return spans_.front().front(); }

  constexpr reference back() const { return spans_.back().back(); }

private:
  std::vector<local_span_type> &
  set_local_views(const std::vector<local_span_type> &spans) {
    my_spans_.resize(0);
    for (auto &&span : spans_) {
      if (span.data().rank == BCL::rank()) {
        my_spans_.push_back(span);
      }
    }
    return my_spans_;
  }

private:
  std::size_t size_ = 0;
  std::vector<local_span_type> spans_;
  joined_view_type joined_view_;
  std::vector<local_span_type> my_spans_;
  joined_view_type local_joined_view_;
};
