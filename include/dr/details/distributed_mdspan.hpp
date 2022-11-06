namespace lib {

template <typename DistributedVector> class distributed_accessor {
public:
  using element_type = typename DistributedVector::element_type;
  using offset_policy = distributed_accessor;
  using reference = typename DistributedVector::reference;
  using data_handle_type = typename DistributedVector::iterator;

  constexpr distributed_accessor() noexcept = default;
  constexpr reference access(data_handle_type p, size_t i) const {
    return p[i];
  }

  constexpr data_handle_type offset(data_handle_type p,
                                    size_t i) const noexcept {
    return p + i;
  }
};

template <typename T, typename Extents, typename Layout = stdex::layout_right,
          typename D = block_cyclic>
class distributed_mdspan {
private:
  using dvector = distributed_vector<T>;
  using dmdspan =
      stdex::mdspan<T, Extents, Layout, distributed_accessor<dvector>>;

public:
#ifdef DR_SPEC
  ///
  using reference = implementation_defined;
  ///
  using value_type = implementation_defined;
  ///
  using size_type = implementation_defined;
  ///
  using difference_type = implementation_defined;
  ///
  using iterator = implementation_defined;
#else
  using reference = typename dmdspan::reference;
  using value_type = typename dvector::value_type;
  using size_type = typename dvector::size_type;
  using difference_type = typename dvector::difference_type;
  using iterator = typename dvector::iterator;
#endif
  ///
  using extents_type = D;

  /// Construct from a distributed_vector with the requested dimesions
  template <typename... Args>
  distributed_mdspan(distributed_vector<T> &dvector, Args... args)
      : extents_(std::forward<Args>(args)...), dvector_(dvector),
        dmdspan_(dvector.begin(), extents_) {
    assert(storage_size(extents_, decomp_.comm().size()) <= dvector.size());
  }

  /// Construct from a distributed_vector with the requested dimesions
  template <typename... Args>
  distributed_mdspan(D decomp, distributed_vector<T> &dvector, Args... args)
      : decomp_(decomp), extents_(std::forward<Args>(args)...),
        dvector_(dvector), dmdspan_(dvector.begin(), extents_) {
    assert(storage_size(extents_, decomp_.comm().size()) <= dvector.size());
  }

  /// first element in layout order
  auto begin() { return dvector_.begin(); }

  /// last element in layout order
  auto end() { return dvector_.end(); }

  /// fence for updates
  void fence() { return dvector_.fence(); }

  /// segments of mdspan
  auto segments() { return dvector_.segments(); }

  /// multidimensional index operator
  template <typename... Args> reference operator()(Args... args) {
    return dmdspan_(std::forward<Args>(args)...);
  }

private:
  D decomp_;
  Extents extents_;
  distributed_vector<T> &dvector_;
  dmdspan dmdspan_;
};

template <typename T, typename Extents, typename Layout = stdex::layout_right,
          typename D = block_cyclic>
class distributed_mdarray {
private:
  using dvector = distributed_vector<T>;
  using dmdspan =
      stdex::mdspan<T, Extents, Layout, distributed_accessor<dvector>>;

public:
#ifdef DR_SPEC
  /// Reference to element
  using reference = implementation_defined;
  /// Reference to element
  using value_type = implementation_defined;
  ///
  using size_type = implementation_defined;
  ///
  using difference_type = implementation_defined;
#else
  using reference = typename dmdspan::reference;
  using value_type = typename dvector::value_type;
  using size_type = typename dvector::size_type;
  using difference_type = typename dvector::difference_type;
  using iterator = typename dvector::iterator;
#endif
  ///
  using extents_type = D;

  /// Construct an mdarray with requested dimensions
  template <typename... Args>
  distributed_mdarray(Args... args)
      : extents_(std::forward<Args>(args)...),
        dvector_(decomp_, storage_size(extents_, decomp_.comm().size())),
        dmdspan_(dvector_.begin(), extents_) {}

  /// Construct an mdarray with requested dimensions
  template <typename... Args>
  distributed_mdarray(D decomp, Args... args)
      : decomp_(decomp), extents_(std::forward<Args>(args)...),
        dvector_(decomp_, storage_size(extents_, decomp_.comm().size())),
        dmdspan_(dvector_.begin(), extents_) {}

  /// first element in layout order
  auto begin() { return dvector_.begin(); }

  /// last element in layout order
  auto end() { return dvector_.end(); }

  /// fence for updates
  void fence() { return dvector_.fence(); }

  /// segments of mdspan
  auto segments() { return dvector_.segments(); }

  /// multidimensional index operator
  template <typename... Args> reference operator()(Args... args) {
    return dmdspan_(std::forward<Args>(args)...);
  }

private:
  D decomp_;
  Extents extents_;
  dvector dvector_;
  dmdspan dmdspan_;
};

} // namespace lib
