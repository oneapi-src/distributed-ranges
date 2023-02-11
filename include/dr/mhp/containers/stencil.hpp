// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

namespace mhp {

class stencil {
public:
  class bounds {
  public:
    bounds(std::size_t s = 0) : prev_(s), next_(s) {}
    bounds(std::size_t prev, std::size_t next) : prev_(prev), next_(next) {}

    auto prev() const { return prev_; }
    auto next() const { return next_; }

  private:
    std::size_t prev_ = 0, next_ = 0;
  };

  stencil(bounds b = bounds()) : bounds_(b) {}

  auto bnds() { return bounds_; }

private:
  bounds bounds_;
};

} // namespace mhp
