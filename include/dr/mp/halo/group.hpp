// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/mp/global.hpp>
#include <dr/mp/sycl_support.hpp>

namespace dr::mp {


  template <typename T, typename Memory = default_memory<T>>
  class index_group {
  public:
    using element_type = T;
    using memory_type = Memory;
    T *buffer = nullptr;
    std::size_t request_index;
    bool receive;
    bool buffered;

    /// Constructor
    index_group(T *data, std::size_t rank,
                const std::vector<std::size_t> &indices, const Memory &memory)
        : memory_(memory), data_(data), rank_(rank) {
      buffered = false;
      for (std::size_t i = 0; i < rng::size(indices) - 1; i++) {
        buffered = buffered || (indices[i + 1] - indices[i] != 1);
      }
      indices_size_ = rng::size(indices);
      indices_ = memory_.template allocate<std::size_t>(indices_size_);
      assert(indices_ != nullptr);
      memory_.memcpy(indices_, indices.data(),
                     indices_size_ * sizeof(std::size_t));
    }

    index_group(const index_group &o)
        : buffer(o.buffer), request_index(o.request_index), receive(o.receive),
          buffered(o.buffered), memory_(o.memory_), data_(o.data_),
          rank_(o.rank_), indices_size_(o.indices_size_), tag_(o.tag_) {
      indices_ = memory_.template allocate<std::size_t>(indices_size_);
      assert(indices_ != nullptr);
      memory_.memcpy(indices_, o.indices_, indices_size_ * sizeof(std::size_t));
    }

    void unpack(const auto &op) {
      T *dpt = data_;
      auto n = indices_size_;
      auto *ipt = indices_;
      auto *b = buffer;
      memory_.offload([=]() {
        for (std::size_t i = 0; i < n; i++) {
          dpt[ipt[i]] = op(dpt[ipt[i]], b[i]);
        }
      });
    }

    void pack() {
      T *dpt = data_;
      auto n = indices_size_;
      auto *ipt = indices_;
      auto *b = buffer;
      memory_.offload([=]() {
        for (std::size_t i = 0; i < n; i++) {
          b[i] = dpt[ipt[i]];
        }
      });
    }

    std::size_t buffer_size() {
      if (buffered) {
        return indices_size_;
      }
      return 0;
    }

    T *data_pointer() {
      if (buffered) {
        return buffer;
      } else {
        return &data_[indices_[0]];
      }
    }

    std::size_t data_size() { return indices_size_; }

    std::size_t rank() { return rank_; }
    auto tag() { return tag_; }

    ~index_group() {
      if (indices_) {
        memory_.template deallocate<std::size_t>(indices_, indices_size_);
        indices_ = nullptr;
      }
    }

  private:
    Memory memory_;
    T *data_ = nullptr;
    std::size_t rank_;
    std::size_t indices_size_;
    std::size_t *indices_;
    halo_tag tag_ = halo_tag::index;
  };

  template <typename T, typename Memory = default_memory<T>> class span_group {
  public:
    using element_type = T;
    using memory_type = Memory;
    T *buffer = nullptr;
    std::size_t request_index = 0;
    bool receive = false;
    bool buffered = false;

    span_group(std::span<T> data, std::size_t rank, halo_tag tag)
        : data_(data), rank_(rank), tag_(tag) {
#ifdef SYCL_LANGUAGE_VERSION
      if (mp::use_sycl() && mp::sycl_mem_kind() == sycl::usm::alloc::shared) {
        buffered = true;
      }
#endif
    }

    /// If span is buffered, push buffer to data
    void unpack() {
      if (buffered) {
        __detail::sycl_copy(buffer, buffer + rng::size(data_), data_.data());
      }
    }

    /// If span is buffered, pull data into buffer
    void pack() {
      if (buffered) {
        __detail::sycl_copy(data_.data(), data_.data() + rng::size(data_), buffer);
      }
    }

    std::size_t buffer_size() { return rng::size(data_); }

    std::size_t data_size() { return rng::size(data_); }

    T *data_pointer() {
      if (buffered) {
        return buffer;
      } else {
        return data_.data();
      }
    }

    std::size_t rank() { return rank_; }

    auto tag() { return tag_; }

  private:
    std::span<T> data_;
    std::size_t rank_;
    halo_tag tag_ = halo_tag::invalid;
  };
}