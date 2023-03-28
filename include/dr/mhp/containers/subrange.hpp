// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once


namespace mhp {

template<typename DM>
class dm_subrange_iterator {
public:
    using value_type = typename DM::value_type;
    using difference_type = typename DM::difference_type;

    dm_subrange_iterator() = delete;
    dm_subrange_iterator(DM * dm, std::pair<std::size_t, std::size_t> first, std::pair<std::size_t, std::size_t> last) { 
        dm_ = dm; 
        first_ = first;
        last_ = last;
        index_ = 0;
    }

    value_type & operator*() { return *(dm_->begin() + find_local_offset(index_)); }

    bool operator==(dm_subrange_iterator &other) { return this->index_ == other.index_; }
    bool operator!=(dm_subrange_iterator &other) { return this->index_ != other.index_; }

    auto operator<=>(const dm_subrange_iterator &other) const noexcept {
        return this->index_ <=> other.index_;
    }
    
    // Only these arithmetics manipulate internal state
    auto &operator-=(difference_type n) { index_ -= n; return *this; }
    auto &operator+=(difference_type n) { index_ += n; return *this; }

    difference_type operator-(const dm_subrange_iterator &other) const noexcept {
        return index_ - other.index_;
    }
    // prefix
    auto &operator++() { *this += 1; return *this; }
    auto &operator--() { *this -= 1; return *this; }

    // postfix
    auto operator++(int) { auto prev = *this; *this += 1; return prev; }
    auto operator--(int) { auto prev = *this; *this -= 1; return prev; }

    auto operator+(difference_type n) const { auto p = *this; p += n; return p; }
    auto operator-(difference_type n) const { auto p = *this; p -= n; return p; }

    // When *this is not first in the expression
    friend auto operator+(difference_type n, const dm_subrange_iterator &other) {
        return other + n;
    }
private:
    /*
     * converts index within subrange (viewed as linear contiguous space)
     * into index within physical segment in dm
     */
    std::size_t const find_local_offset(size_t index) {
        std::size_t ind_rows, ind_cols;
        std::size_t offset;

        ind_rows = index / (last_.first - first_.first);
        ind_cols = index % (last_.first - first_.first);

        offset = first_.first * dm_->shape()[0] + first_.second;
        offset += ind_rows * dm_->shape()[0] + ind_cols;

        return offset / dm_->segsize();
    }

private:
    DM * dm_;
    std::pair<std::size_t, std::size_t> first_;
    std::pair<std::size_t, std::size_t> last_;

    size_t index_ = 0;
};

template <typename DM> class subrange {

public:
    using iterator = dm_subrange_iterator<DM>;
    using value_type = typename DM::value_type;

    subrange(DM & dm, std::pair<std::size_t, std::size_t> first, std::pair<std::size_t, std::size_t> last) {
        dm_ = &dm; 
        first_ = first;
        last_ = last;

        subrng_size_ = (last.first - first.first) * (last.second - first.second);
    }

    iterator begin() { return iterator(dm_, first_, last_); }
    iterator end() { return iterator(0) + subrng_size_; }

    value_type front() { return *(begin()); }
    value_type back() { return *(end()); }
    

    auto &halo() const { return dm_->halo(); }
    
    auto segments() const { return dm_->segments(); }

private:
    DM * dm_;
    std::pair<std::size_t, std::size_t> first_;
    std::pair<std::size_t, std::size_t> last_;

    std::size_t subrng_size_ = 0;

}; // class subrange

} // namespace mhp
