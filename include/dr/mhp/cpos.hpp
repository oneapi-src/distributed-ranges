// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

namespace mhp {

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
