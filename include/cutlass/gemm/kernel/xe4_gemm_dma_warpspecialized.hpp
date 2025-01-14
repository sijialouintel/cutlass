#pragma once

#include <sycl/sycl.hpp>
#include <cute/tensor.hpp>

#include "cute/arch/copy_xe4_dma.hpp"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/epilogue/collective/collective_epilogue.hpp"

namespace cutlass::gemm::kernel {

using namespace cute;
using namespace sycl;
using namespace cute::xe4;

template <
  class ProblemShape_,
  class CollectiveMainloop_,
  class CollectiveEpilogue_,
  class TileScheduler_
>
class GemmUniversal<
  ProblemShape_,
  CollectiveMainloop_,
  CollectiveEpilogue_,
  TileScheduler_,
  cute::enable_if_t<cute::is_base_of_v<cutlass::gemm::KernelXe4WarpSpecialized, typename CollectiveMainloop_::DispatchPolicy::Schedule>>>
{
public:
  //
  // Type Aliases
  //
  using ProblemShape = ProblemShape_;
  static_assert(cute::rank(ProblemShape{}) == 3 or cute::rank(ProblemShape{}) == 4,
    "ProblemShape{} should be <M,N,K> or <M,N,K,L>");
  // Mainloop derived types
  using CollectiveMainloop = CollectiveMainloop_;
  using TileShape = typename CollectiveMainloop::TileShape;
  using TiledMma  = typename CollectiveMainloop::TiledMma;
  using ArchTag   = typename CollectiveMainloop::ArchTag;
  using ElementA  = typename CollectiveMainloop::ElementA;
  using StrideA   = typename CollectiveMainloop::StrideA;
  using SmemLayoutA = typename CollectiveMainloop::SmemLayoutA;
  using ElementB  = typename CollectiveMainloop::ElementB;
  using StrideB   = typename CollectiveMainloop::StrideB;
  using SmemLayoutB = typename CollectiveMainloop::SmemLayoutB;
  using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
  using ElementAccumulator = typename CollectiveMainloop::ElementAccumulator;
  using SmemLayoutC = decltype(make_layout(take<0,2>(TileShape{}), GenRowMajor{}));
  using ClusterShape = typename DispatchPolicy::ClusterShape;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;

  using TileSchedulerTag = TileScheduler_;
  using TileScheduler = typename detail::TileSchedulerSelector<
    TileSchedulerTag, ArchTag, TileShape, ClusterShape>::Scheduler;
  using TileSchedulerArguments = typename TileScheduler::Arguments;
  using TileSchedulerParams = typename TileScheduler::Params;

  struct SharedStorage
  {
    using MainloopTensorStorage = typename CollectiveMainloop::TensorStorage;
    using EpilogueTensorStorage = typename CollectiveEpilogue::TensorStorage;

    struct TensorStorage
    {
      MainloopTensorStorage mainloop;
      EpilogueTensorStorage epilogue;
    } tensors;
  };

  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  struct GroupInfo {
    uint32_t subgroup_size = 0;
    uint32_t mainloop_subgroup_num = 0;
    uint32_t epilogue_subgroup_num = 0;
  };

  // Device side arguments
  struct Arguments {
    GroupInfo group_info;
    ProblemShape problem_shape;
    MainloopArguments mainloop;
    EpilogueArguments epilogue;
  };

  // Kernel entry point API
  struct Params {
    GroupInfo group_info;
    ProblemShape problem_shape;
    MainloopParams mainloop;
    EpilogueParams epilogue;
    TileSchedulerParams scheduler;
  };

  static
  Params
  to_underlying_arguments(Arguments const& args, void* workspace) {
    (void) workspace;

    auto scheduler_args = typename TileScheduler::Arguments {
      {CollectiveMainloop::SlmBytesA, CollectiveMainloop::SlmBytesB}
    };

    return {
      args.group_info,
      args.problem_shape,
      CollectiveMainloop::to_underlying_arguments(args.problem_shape, args.mainloop),
      CollectiveEpilogue::to_underlying_arguments(args.problem_shape, args.epilogue, nullptr),
      TileScheduler::to_underlying_arguments(args.problem_shape, TileShape{}, ClusterShape{}, scheduler_args)
    };
  }

  CUTLASS_DEVICE
  void
  operator()(Params const& params, sycl::nd_item<3> item) {
    auto& group_info = params.group_info;
    auto& problem_shape = params.problem_shape;

    auto ptr = alloc_slm_buffer<uint8_t, SharedStorageSize>(item.get_group());
    auto& shared_tensors = reinterpret_cast<SharedStorage*>(ptr)->tensors;

    enum class SubGroupRole {
      Producer = 0,
      Consumer = 1,
      Store = 2,
      Epilogue = 3,
      NonParticipant
    };

    using MainloopPipeline = typename CollectiveMainloop::MainloopPipeline;
    using EpiloguePipeline = typename CollectiveEpilogue::PostOpPipeline;
    using StorePipeline = typename CollectiveEpilogue::StorePipeline;

    using MainloopPipelineState = typename CollectiveMainloop::PipelineState;
    using EpiloguePipelineState = typename CollectiveEpilogue::PostOpPipelineState;
    using StorePipelineState = typename CollectiveEpilogue::StorePipelineState;

    uint32_t local_id = item.get_local_linear_id();
    constexpr uint32_t abar_count = 2 * (MainloopPipeline::Stages + EpiloguePipeline::Stages + StorePipeline::Stages);

    auto abar_base = allocate_abar<0, abar_count>();
    MainloopPipeline mainloop_pipeline(abar_base, local_id);
    auto abar_epilogue_base = abar_base + 2 * MainloopPipeline::Stages;
    EpiloguePipeline epilogue_pipeline(abar_epilogue_base, local_id);
    auto abar_store_base = abar_epilogue_base + 2 * EpiloguePipeline::Stages;
    auto epilogue_thread_count = group_info.epilogue_subgroup_num * group_info.subgroup_size;
    StorePipeline epilogue_store_pipeline(abar_store_base, local_id, epilogue_thread_count, 1);

    CollectiveMainloop collective_mainloop;
    CollectiveEpilogue collective_epilogue(params.epilogue);

    auto mainloop_pipe_producer_state = cutlass::xe4::make_producer_start_state<MainloopPipeline>();
    auto mainloop_pipe_consumer_state = MainloopPipelineState{};

    auto epilogue_pipe_producer_state = cutlass::xe4::make_producer_start_state<EpiloguePipeline>();
    auto epilogue_pipe_consumer_state = EpiloguePipelineState{};

    auto store_pipe_producer_state = cutlass::xe4::make_producer_start_state<StorePipeline>();
    auto store_pipe_consumer_state = StorePipelineState{};

    auto K = get<2>(problem_shape);
    auto wg_k = get<2>(TileShape{});
    uint32_t k_tile_count = (K + wg_k -1) / wg_k;

    using SmemLayoutD = typename CollectiveEpilogue::SmemLayoutD;
    auto tensorD = make_tensor(shared_tensors.epilogue.smem_D.data(), SmemLayoutD{});

    auto warp_group_role = SubGroupRole::NonParticipant;
    if (local_id == 0) {
      warp_group_role = SubGroupRole::Producer;
    } else if (local_id == group_info.subgroup_size) {
      warp_group_role = SubGroupRole::Consumer;
    } else if (local_id == 2 * group_info.subgroup_size) {
      warp_group_role = SubGroupRole::Store;
    } else if (local_id >= group_info.mainloop_subgroup_num * group_info.subgroup_size) {
      warp_group_role = SubGroupRole::Epilogue;
    }

    auto cluster_wait_fn = [&] () {
      // We need this to guarantee that the Pipeline init is visible
      // To all producers and consumer thread blocks in the Cluster
      if constexpr (size(ClusterShape{}) > 1) {
        cbar_arrive();
        return [] () { cbar_wait(); };
      }
      else {
        item.barrier(access::fence_space::local_space);
        return [] () {}; // do nothing
      }
    } ();

    // Wait for all thread blocks in the Cluster
    cluster_wait_fn();

    auto load_inputs = collective_mainloop.load_init(problem_shape, params.mainloop);
    static_assert(cute::tuple_size_v<decltype(load_inputs)> >= 2, "Output of load_init must have at least two elements (A, B)");

    auto group = item.get_group();
    uint32_t group_id_m = group.get_group_id(1);
    uint32_t group_id_n = group.get_group_id(2);
    uint32_t group_range_m = group.get_group_range(1);
    uint32_t group_range_n = group.get_group_range(2);

    auto scheduler = TileScheduler::make_scheduler(params.scheduler, {group_id_m, group_id_n}, {group_range_m, group_range_n});
    auto work_tile_info = scheduler.initial_work_tile_info();

    auto coop_ids = params.scheduler.coop_ids;
    auto cluster_masks = params.scheduler.cluster_masks;

    if (warp_group_role == SubGroupRole::Producer) {
      while (work_tile_info.is_valid()) {
        auto m_coord = work_tile_info.M_idx;
        auto n_coord = work_tile_info.N_idx;
        auto l_coord = work_tile_info.L_idx;
        auto blk_coord = make_coord(m_coord, n_coord, l_coord);

        collective_mainloop.load(params.mainloop, mainloop_pipeline, mainloop_pipe_producer_state, load_inputs, blk_coord, k_tile_count, local_id, coop_ids, cluster_masks, shared_tensors.mainloop);

        // Get next work tile
        scheduler.advance_to_next_work();
        work_tile_info = scheduler.get_current_work();
      } // Scheduler work fetch loop
    } else if (warp_group_role == SubGroupRole::Consumer) {
      while (work_tile_info.is_valid()) {
        collective_mainloop.mma(mainloop_pipeline, mainloop_pipe_consumer_state, epilogue_store_pipeline, store_pipe_producer_state, epilogue_pipeline, epilogue_pipe_producer_state, tensorD, k_tile_count, local_id, cluster_masks, shared_tensors.mainloop);

        // Get next work tile
        scheduler.advance_to_next_work();
        work_tile_info = scheduler.get_current_work();
      } // Scheduler work fetch loop
    } else if (warp_group_role == SubGroupRole::Epilogue) {
      uint32_t work_id = local_id - group_info.mainloop_subgroup_num * group_info.subgroup_size;
      while (work_tile_info.is_valid()) {
        collective_epilogue(epilogue_store_pipeline, store_pipe_producer_state, epilogue_pipeline, epilogue_pipe_consumer_state, shared_tensors.epilogue, work_id);

        // Get next work tile
        scheduler.advance_to_next_work();
        work_tile_info = scheduler.get_current_work();
      } // Scheduler work fetch loop
    } else if (warp_group_role == SubGroupRole::Store) {
      while (work_tile_info.is_valid()) {
        auto m_coord = work_tile_info.M_idx;
        auto n_coord = work_tile_info.N_idx;
        auto l_coord = work_tile_info.L_idx;
        auto blk_coord = make_coord(m_coord, n_coord, l_coord);

        collective_epilogue.store(epilogue_store_pipeline, store_pipe_consumer_state, problem_shape, blk_coord, shared_tensors.epilogue);

        // Get next work tile
        scheduler.advance_to_next_work();
        work_tile_info = scheduler.get_current_work();
      } // Scheduler work fetch loop
    }
  }
};

}
