// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "segment.hpp"

#pragma once

namespace dr::mp {

template <typename DV> class dual_dv_segment_iterator : public dv_segment_iterator<DV> {
protected:
  DV::backend_type& backend() override {
    return dv_->backends[0];
  }
};

template <typename DV> class dual_dv_segment : public dv_segment<DV> {
private:
  using iterator = dual_dv_segment_iterator<DV>;

public:
  using difference_type = std::ptrdiff_t;
  dual_dv_segment() = default;
  dual_dv_segment(DV *dv, std::size_t segment_index, std::size_t size,
                  std::size_t reserved) 
                  : dv_segment<DV>(dv, segment_index, size, reserved) {
  }

  bool is_compute() const { return _is_compute; }

  void swap_state() { _is_compute = !_is_compute; }

private:
  bool _is_compute = true;
}; // dual_dv_segment

} // namespace dr::mp
