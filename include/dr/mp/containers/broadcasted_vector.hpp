// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

namespace dr::mp {

    
template <typename T, typename Allocator = dr::mp::__detail::allocator<T>>
class broadcasted_vector {
    public:
    broadcasted_vector() = default;

    template <rng::input_range R>
    void broadcast_data(std::size_t data_size, std::size_t root, R root_data, dr::communicator comm) {
        if (_data != nullptr) {
            destroy_data();
        }
        _data_size = data_size;
        _data = alloc.allocate(_data_size);
        if (comm.rank() == root) {
            if (use_sycl()) {
                __detail::sycl_copy(std::to_address(root_data.begin()), std::to_address(root_data.end()), _data);
            }
            else {
                rng::copy(root_data.begin(), root_data.end(), _data);
            }
        }
        comm.bcast(_data, sizeof(T) * _data_size, root);
    }
    
    void destroy_data() {
        alloc.deallocate(_data, _data_size);
        _data_size = 0;
        _data = nullptr;
    }

    T& operator[](std::size_t index) {
        return _data[index];
    }
    
    T* broadcasted_data() {
        return _data;
    }

    auto size() {
        return _data_size;
    }
    private:
    T* _data = nullptr;
    std::size_t _data_size = 0;
    Allocator alloc;
};
}