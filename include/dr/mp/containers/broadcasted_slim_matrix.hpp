// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

namespace dr::mp {

    
template <typename T, typename Allocator = dr::mp::__detail::allocator<T>>
class broadcasted_slim_matrix {
    public:
    broadcasted_slim_matrix() = default;

    void broadcast_data(std::size_t height, std::size_t width, std::size_t root, T** root_data, dr::communicator comm) {
        if (_data != nullptr) {
            destroy_data();
        }
        _data_size = height * width;
        _height = height;
        _width = width;
        _data = alloc.allocate(_data_size);
        if (comm.rank() == root) {
            for (auto i = 0; i < width; i++) {
              rng::copy(root_data[i], root_data[i] + height, _data);
            }
        }
        comm.bcast(_data, sizeof(T) * _data_size, root);
    }
    
    template <rng::input_range R>
    void broadcast_data(std::size_t height, std::size_t width, std::size_t root, R root_data, dr::communicator comm) {
        if (_data != nullptr) {
            destroy_data();
        }
        _data_size = height * width;
        _height = height;
        _width = width;
        _data = alloc.allocate(_data_size);
        if (comm.rank() == root) {
            rng::copy(root_data.begin(), root_data.end(), _data);
        }
        comm.bcast(_data, sizeof(T) * _data_size, root);
    }

    void destroy_data() {
        alloc.deallocate(_data, _data_size);
        _data_size = 0;
        _data = nullptr;
    }

    T* operator[](std::size_t index) {
        return _data + _height * index;
    }

    T* broadcasted_data() {
        return _data;
    }
    auto width() {
        return _width;
    }
    private:
    T* _data = nullptr;
    std::size_t _data_size = 0;
    std::size_t _width = 0;
    std::size_t _height = 0;

    Allocator alloc;
};
}