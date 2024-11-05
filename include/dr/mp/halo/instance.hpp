// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/mp/global.hpp>
#include <dr/mp/sycl_support.hpp>
#include <dr/mp/halo/halo.hpp>
#include <dr/mp/halo/group.hpp>

namespace dr::mp {
template<typename T, typename Memory>
using unstructured_halo_impl = halo_impl <index_group<T, Memory>>;

template<typename T, typename Memory = default_memory <T>>
class unstructured_halo : public unstructured_halo_impl<T, Memory> {
public:
  using group_type = index_group<T, Memory>;
  using index_map = std::pair <std::size_t, std::vector<std::size_t>>;

  ///
  /// Constructor
  ///
  unstructured_halo(communicator comm, T *data,
                    const std::vector <index_map> &owned,
                    const std::vector <index_map> &halo,
                    const Memory &memory = Memory())
      : unstructured_halo_impl<T, Memory>(
      comm, make_groups(comm, data, owned, memory),
      make_groups(comm, data, halo, memory), memory) {}

private:
  static std::vector <group_type> make_groups(communicator comm, T *data,
                                              const std::vector <index_map> &map,
                                              const Memory &memory) {
    std::vector <group_type> groups;
    for (auto const &[rank, indices]: map) {
      groups.emplace_back(data, rank, indices, memory);
    }
    return groups;
  }
};

template<typename T, typename Memory>
using span_halo_impl = halo_impl <span_group<T, Memory>>;

template<typename T, typename Memory = default_memory <T>>
class span_halo : public span_halo_impl<T, Memory> {
public:
  using group_type = span_group<T, Memory>;

  span_halo() : span_halo_impl<T, Memory>(communicator(), {}, {}) {}

  span_halo(communicator comm, T *data, std::size_t size, halo_bounds hb)
      : span_halo_impl<T, Memory>(comm, owned_groups(comm, {data, size}, hb),
                                  halo_groups(comm, {data, size}, hb)) {
    check(size, hb);
  }

  span_halo(communicator comm, std::span <T> span, halo_bounds hb)
      : span_halo_impl<T, Memory>(comm, owned_groups(comm, span, hb),
                                  halo_groups(comm, span, hb)) {}

private:
  void check(auto size, auto hb) {
    assert(size >= hb.prev + hb.next + std::max(hb.prev, hb.next));
  }

  static std::vector <group_type>
  owned_groups(communicator comm, std::span <T> span, halo_bounds hb) {
    std::vector <group_type> owned;
    DRLOG("owned groups {}/{} first/last", comm.first(), comm.last());
    if (hb.next > 0 && (hb.periodic || !comm.first())) {
      owned.emplace_back(span.subspan(hb.prev, hb.next), comm.prev(),
                         halo_tag::reverse);
    }
    if (hb.prev > 0 && (hb.periodic || !comm.last())) {
      owned.emplace_back(
          span.subspan(rng::size(span) - (hb.prev + hb.next), hb.prev),
          comm.next(), halo_tag::forward);
    }
    return owned;
  }

  static std::vector <group_type>
  halo_groups(communicator comm, std::span <T> span, halo_bounds hb) {
    std::vector <group_type> halo;
    if (hb.prev > 0 && (hb.periodic || !comm.first())) {
      halo.emplace_back(span.first(hb.prev), comm.prev(), halo_tag::forward);
    }
    if (hb.next > 0 && (hb.periodic || !comm.last())) {
      halo.emplace_back(span.last(hb.next), comm.next(), halo_tag::reverse);
    }
    return halo;
  }
};
}