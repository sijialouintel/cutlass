#pragma once

#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/gemm_coord.hpp"

namespace cutlass::gemm::kernel::detail {

///////////////////////////////////////////////////////////////////////////////

class PersistentTileSchedulerXe4 {
public:
  struct Arguments {
    cute::tuple<uint32_t, uint32_t> slm_bytes;
  };

  struct Params {
    cute::tuple<int, int, int> problem_blocks_range;
    cute::tuple<int, int, int> problem_blocks_shape;
    cute::tuple<uint32_t, uint32_t> cluster_shape;
    cute::tuple<uint32_t, uint32_t> cluster_masks;
    cute::tuple<uint32_t, uint32_t> coop_set_ids;
    cute::tuple<uint32_t, uint32_t> coop_ids;
  };

  template <class ProblemShapeMNKL, class TileShape, class ClusterShape>
  static Params
  to_underlying_arguments(
      ProblemShapeMNKL problem_shape_mnkl,
      TileShape tile_shape,
      ClusterShape cluster_shape,
      Arguments const& arguments,
      [[maybe_unused]] void* workspace=nullptr) {

    // We only need the tile and cluster shape during scheduler setup, so let FTAD do the magic
    static_assert(cute::is_static<TileShape>::value);
    static_assert(cute::is_static<ClusterShape>::value);

    auto problem_shape_mnl = cute::select<0, 1, 3>(problem_shape_mnkl);
    auto problem_blocks_range = cute::ceil_div(flatten(problem_shape_mnl), flatten(tile_shape));

    auto [cluster_size_m, cluster_size_n, _] = cluster_shape;
    auto problem_blocks_m = cute::round_up(cute::get<0>(problem_blocks_range), cluster_size_m);
    auto problem_blocks_n = cute::round_up(cute::get<1>(problem_blocks_range), cluster_size_n);
    auto problem_blocks_shape = cute::make_shape(problem_blocks_m, problem_blocks_n, cute::get<2>(problem_blocks_range));

    if (cluster_size_n * cluster_size_m == 1) {
      return {
        problem_blocks_range,
        problem_blocks_shape,
        {cluster_size_m, cluster_size_n},
        {0, 0},
        {0, 0},
        {0, 0}
      };
    }

    uint32_t cluster_wgid_x = get_cluster_wgid<0>();
    uint32_t cluster_wgid_y = get_cluster_wgid<1>();

    uint32_t cluster_mask_a_ = 0;
    uint32_t cluster_mask_b_ = 0;
    uint32_t coop_id_a_ = cluster_wgid_x;
    uint32_t coop_id_b_ = cluster_wgid_y;
    uint32_t coop_set_id_a_ = cluster_wgid_y;
    uint32_t coop_set_id_b_ = cluster_wgid_x;

    uint32_t coop_num_a = cluster_size_n;
    uint32_t coop_num_b = cluster_size_m;
    uint32_t multicast_size_a = get<0>(arguments.slm_bytes) / coop_num_a;
    uint32_t multicast_size_b = get<1>(arguments.slm_bytes) / coop_num_b;

    if (multicast_size_a >= multicast_size_b) {
      cluster_mask_a_ = ((1u << coop_num_a) - 1) << (coop_set_id_a_ * coop_num_a); //0011
      uint32_t cluster_mask_b_base = 1u << coop_set_id_b_;
      #pragma unroll
      for (uint32_t i = 0; i < coop_num_b; i++) {
        cluster_mask_b_ |= cluster_mask_b_base << (i * coop_num_a);
      }
    } else {
      uint32_t cluster_wgid = cluster_wgid_y * cluster_size_n + cluster_wgid_x;
      cluster_wgid_x = cluster_wgid % coop_num_b;
      cluster_wgid_y = cluster_wgid / coop_num_b;
      coop_set_id_a_ = cluster_wgid_x;
      coop_set_id_b_ = cluster_wgid_y;
      coop_id_a_ = cluster_wgid_y;
      coop_id_b_ = cluster_wgid_x;

      cluster_mask_b_ = ((1u << coop_num_b) - 1) << (coop_set_id_b_ * coop_num_b); //0011
      uint32_t cluster_mask_a_base = 1u << coop_set_id_a_;
      #pragma unroll
      for (uint32_t i = 0; i < coop_num_a; i++) {
        cluster_mask_a_ |= cluster_mask_a_base << (i * coop_num_b);
      } //0101
    }

    return {
      problem_blocks_range,
      problem_blocks_shape,
      {cluster_size_m, cluster_size_n},
      {cluster_mask_a_, cluster_mask_b_},
      {coop_set_id_a_, coop_set_id_b_},
      {coop_id_a_, coop_id_b_}
    };
  }

  struct WorkTileInfo {
    int32_t M_idx = 0;
    int32_t N_idx = 0;
    int32_t L_idx = 0;
    bool is_valid_tile = false;

    CUTLASS_HOST_DEVICE
    bool
    is_valid() const {
      return is_valid_tile;
    }

    CUTLASS_HOST_DEVICE
    static WorkTileInfo
    invalid_work_tile() {
      return {-1, -1, -1, false};
    }

    CUTLASS_HOST_DEVICE
    bool
    is_final_split(uint32_t k_tiles_per_output_tile) const {
      return true;
    }

    CUTLASS_HOST_DEVICE
    int32_t
    reduction_subtile_idx() const {
      return -1;
    }
  };

  template <typename CoordTensor>
  class Impl {
    public:
      Impl(Params const& params, CoordTensor const& coord_tensor, cute::tuple<uint32_t, uint32_t> wg_id)
        : params_(params), wgid_(wg_id), coord_tensor_(coord_tensor) {}

      CUTLASS_DEVICE
      WorkTileInfo initial_work_tile_info() {
        return get_current_work();
      }

      CUTLASS_DEVICE
      WorkTileInfo get_current_work() const {
        if (iter_id_ >= size<1>(coord_tensor_)) {
          return WorkTileInfo::invalid_work_tile();
        }

        const auto& cluster_local_id = params_.coop_set_ids;
        const auto cluster_id = cute::transform(wgid_, params_.cluster_shape, [](auto x, auto y) { return x / y; });
        auto [coord_m, coord_n, coord_l] = coord_tensor_(cute::make_coord(cluster_local_id, cluster_id), iter_id_);

        return {
          static_cast<int32_t>(coord_m),
          static_cast<int32_t>(coord_n),
          static_cast<int32_t>(coord_l),
          true
        };
      }

      CUTLASS_DEVICE
      void
      advance_to_next_work(uint32_t advance_count = 1) {
        ++iter_id_;
      }

    private:
      Params params_;
      uint32_t iter_id_ {0};
      CoordTensor coord_tensor_;
      cute::tuple<uint32_t, uint32_t> wgid_;
  };

  static auto make_scheduler(Params const& params, cute::tuple<uint32_t, uint32_t> wg_id, cute::tuple<uint32_t, uint32_t> group_range) {
    auto wg_gride_layout = make_layout(group_range, make_stride(cute::E<0>{}, cute::E<1>{}));
    auto tiled_wg_gride_layout = zipped_divide(wg_gride_layout, params.cluster_shape);

    auto problem_blocks_layout = cute::make_layout(params.problem_blocks_shape, cute::make_stride(cute::E<0>{}, cute::E<1>{}, cute::E<2>{}));
    auto tiled_problem_blocks_layout = zipped_divide(problem_blocks_layout, group_range);

    auto coord_tensor_layout = replace<0>(tiled_problem_blocks_layout, tiled_wg_gride_layout);
    auto coord_tensor = cute::make_tensor(cute::make_inttuple_iter(0,0), coord_tensor_layout);

    return Impl<decltype(coord_tensor)>{params, coord_tensor, wg_id};
  }
};

}
