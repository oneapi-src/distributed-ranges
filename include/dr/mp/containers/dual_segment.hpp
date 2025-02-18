// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "segment.hpp"

#pragma once

namespace dr::mp {

// template <typename DV> 
// class dual_dv_segment_iterator : public dv_segment_iterator<DV> {
// public:
//   dual_dv_segment_iterator() = default;
//   dual_dv_segment_iterator(DV *dv, std::size_t segment_index, std::size_t index)
//     : dv_segment_iterator<DV>(dv, segment_index, index) {
//   }

//   auto rank() const {
//     std::cout << "rank(): segment_index_ == " << this->segment_index_ << "\n";

//     if (this->segment_index_ < default_comm().size()) {
//       return this->segment_index_;
//     }

//     return 2 * default_comm().size() - this->segment_index_ - 1;
//   }
// };

template <typename DV>
class dual_dv_segment : public dv_segment<DV> {
private:
  using iterator = dv_segment_iterator<DV>;

public:
  using difference_type = std::ptrdiff_t;
  dual_dv_segment() = default;
  dual_dv_segment(DV *dv, std::size_t segment_index, std::size_t size,
                  std::size_t reserved) 
                  : dv_segment<DV>(dv, segment_index, size, reserved) {
  }

  // auto size() const {
  //   assert(this->dv_ != nullptr);
  //   return this->size_;
  // }

  // auto begin() const { return iterator(this->dv_, this->segment_index_, 0); }
  // auto end() const { return begin() + size(); }
  // auto reserved() const { return this->reserved_; }

  // auto operator[](difference_type n) const { return *(begin() + n); }

  bool is_local() const { 
    return this->segment_index_ == default_comm().rank()
      || this->segment_index_ == 2 * default_comm().size() - default_comm().rank() - 1;
  }

  bool is_compute() const { return _is_compute; }

  void swap_state() { _is_compute = !_is_compute; }

private:
  bool _is_compute = true;
}; // dual_dv_segment

} // namespace dr::mp
