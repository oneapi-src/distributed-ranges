// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

namespace mhp {

class halo_bounds {
public:
  halo_bounds(std::size_t s = 0) : prev_(s), next_(s) {}
  halo_bounds(std::size_t prev, std::size_t next) : prev_(prev), next_(next) {}

  auto prev() const { return prev_; }
  auto next() const { return next_; }

private:
  friend fmt::formatter<mhp::halo_bounds>;
  std::size_t prev_ = 0, next_ = 0;
};

class stencil {
public:
  stencil(halo_bounds hb = halo_bounds()) : halo_bounds_(hb) {}
  stencil(std::size_t b) : halo_bounds_(b) {}

  auto bounds() { return halo_bounds_; }

private:
  friend fmt::formatter<mhp::stencil>;
  halo_bounds halo_bounds_;
};

} // namespace mhp

template <> struct fmt::formatter<mhp::halo_bounds> : formatter<string_view> {
  template <typename FmtContext>
  auto format(mhp::halo_bounds hb, FmtContext &ctx) {
    return format_to(ctx.out(), "prev: {} next: {}", hb.prev_, hb.next_);
  }
};

template <> struct fmt::formatter<mhp::stencil> : formatter<string_view> {
  template <typename FmtContext> auto format(mhp::stencil s, FmtContext &ctx) {
    return format_to(ctx.out(), "bounds: ({})", s.halo_bounds_);
  }
};
