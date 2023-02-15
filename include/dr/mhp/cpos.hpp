// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

namespace mhp {

namespace internal {

template <typename Iter>
concept has_fence_method = requires(Iter i) {
                             { i.fence() };
                           };

template <typename Iter>
concept has_fence_adl = requires(Iter &iter) {
                          { fence_(iter) };
                        };

void fence_(lib::internal::is_zip_iterator auto zi) {
  auto fence = [](auto &&...refs) { (((&refs).fence()), ...); };
  std::apply(fence, *zi);
}

struct fence_fn_ {
  template <std::forward_iterator Iter> void operator()(Iter iter) const {
    static_assert(has_fence_method<Iter> || has_fence_adl<Iter>);
    if constexpr (has_fence_method<Iter>) {
      iter.fence();
    } else if constexpr (has_fence_adl<Iter>) {
      fence_(iter);
    }
  }
};

} // namespace internal

inline constexpr auto fence = mhp::internal::fence_fn_{};

namespace internal {

void barrier_(lib::internal::is_zip_iterator auto zi) {
  auto barrier = [](auto &&...refs) { (((&refs).barrier()), ...); };
  std::apply(barrier, *zi);
}

template <typename Iter>
concept has_barrier_method = requires(Iter i) {
                               { i.barrier() };
                             };

template <typename Iter>
concept has_barrier_adl = requires(Iter &iter) {
                            { barrier_(iter) };
                          };
struct barrier_fn_ {
  template <std::forward_iterator Iter> void operator()(Iter iter) const {
    static_assert(has_barrier_method<Iter> || has_barrier_adl<Iter>);
    if constexpr (has_barrier_method<Iter>) {
      iter.barrier();
    } else if constexpr (has_barrier_adl<Iter>) {
      barrier_(iter);
    }
  }
};

} // namespace internal

inline constexpr auto barrier = mhp::internal::barrier_fn_{};

namespace internal {

void halo_exchange_(lib::internal::is_zip_iterator auto zi) {
  auto halo_exchange = [](auto &&...refs) { (((&refs).halo_exchange()), ...); };
  std::apply(halo_exchange, *zi);
}

template <typename Iter>
concept has_halo_exchange_method = requires(Iter i) {
                                     { i.halo_exchange() };
                                   };

template <typename Iter>
concept has_halo_exchange_adl = requires(Iter &iter) {
                                  { halo_exchange_(iter) };
                                };
struct halo_exchange_fn_ {
  template <std::forward_iterator Iter> void operator()(Iter iter) const {
    static_assert(has_halo_exchange_method<Iter> ||
                  has_halo_exchange_adl<Iter>);
    if constexpr (has_halo_exchange_method<Iter>) {
      iter.halo_exchange();
    } else if constexpr (has_halo_exchange_adl<Iter>) {
      halo_exchange_(iter);
    }
  }
};

} // namespace internal

inline constexpr auto halo_exchange = mhp::internal::halo_exchange_fn_{};

} // namespace mhp
