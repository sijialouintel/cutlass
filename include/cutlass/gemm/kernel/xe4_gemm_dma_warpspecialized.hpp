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

  // Pipeline and pipeline state types
  using MainloopPipeline = typename CollectiveMainloop::MainloopPipeline;
  using MainloopPipelineState = typename CollectiveMainloop::MainloopPipelineState;

  using EpiStorePipeline = typename CollectiveEpilogue::StorePipeline;
  using EpiStorePipelineState = typename CollectiveEpilogue::StorePipelineState;

  using AccumulatorPipeline = typename CollectiveEpilogue::AccumulatorPipeline;
  using AccumulatorPipelineState = typename CollectiveEpilogue::AccumulatorPipelineState;

  // Kernel level shared memory storage
  struct SharedStorage
  {
    struct TensorStorage
    {
      using MainloopTensorStorage = typename CollectiveMainloop::TensorStorage;
      using EpilogueTensorStorage = typename CollectiveEpilogue::TensorStorage;

      MainloopTensorStorage mainloop;
      EpilogueTensorStorage epilogue;
    } tensors;

    struct PipelineStorage {
      using MainloopPipelineStorage = typename MainloopPipeline::SharedStorage;
      using AccumulatorPipelineStorage = typename AccumulatorPipeline::SharedStorage;
      using EpiStorePipelineStorage = typename EpiStorePipeline::SharedStorage;

      MainloopPipelineStorage mainloop;
      AccumulatorPipelineStorage accumulator;
      EpiStorePipelineStorage epi_store;
    } pipelines;
  };

  static constexpr int TensorStorageSize = sizeof(typename SharedStorage::TensorStorage);
  static constexpr int PipelineStorageSize = sizeof(typename SharedStorage::PipelineStorage);

  struct GroupInfo {
    uint32_t subgroup_size = 0;
    uint32_t mainloop_subgroup_num = 0;
    uint32_t epilogue_subgroup_num = 0;
  };

  // Host facing host arguments
  struct Arguments {
    GroupInfo group_info;
    ProblemShape problem_shape;
    MainloopArguments mainloop;
    EpilogueArguments epilogue;
  };

  // Kernel device entry point API
  struct Params {
    GroupInfo group_info;
    ProblemShape problem_shape;
    MainloopParams mainloop;
    EpilogueParams epilogue;
    TileSchedulerParams scheduler;
  };

  enum class WarpCategory : int32_t {
    MMA          = 0,
    Sched        = 1,
    MainloopLoad = 2,
    EpilogueStore = 3,
    Epilogue     = 4
  };

  struct IsParticipant {
    uint32_t mma       = false;
    uint32_t sched     = false;
    uint32_t main_load = false;
    uint32_t epi_store  = false;
    uint32_t epilogue  = false;
  };

  //
  // Methods
  //

  // Convert to underlying arguments.
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

    // Warp specialization thread count per threadblock
    uint32_t SubGroupSize            = group_info.subgroup_size;
    uint32_t NumSchedThreads         = SubGroupSize; // 1 subgroup
    uint32_t NumMMAThreads           = SubGroupSize; // 1 subgroup
    uint32_t NumMainloopLoadThreads  = SubGroupSize; // 1 subgroup
    uint32_t NumEpilogueStoreThreads = SubGroupSize; // 1 subgroup
    uint32_t NumEpilogueThreads      = group_info.epilogue_subgroup_num * SubGroupSize;

    // Separate out problem shape for convenience
    // Optionally append 1s until problem shape is rank-4 in case its is only rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(params.problem_shape, Int<1>{});
    auto [M,N,K,L] = problem_shape_MNKL;

   // Account for more than one epilogue warp
    uint32_t local_id = item.get_local_linear_id();
    uint32_t warp_idx = get_sg_id();
    WarpCategory warp_category = warp_idx < static_cast<int>(WarpCategory::Epilogue) ? WarpCategory(warp_idx)
                                                                                     : WarpCategory::Epilogue;

    uint32_t lane_predicate = local_id % group_info.subgroup_size == 0;

    auto ptr = alloc_slm_buffer<uint8_t, TensorStorageSize>(item.get_group());
    auto& shared_tensors = *reinterpret_cast<typename SharedStorage::TensorStorage*>(ptr);

    enum class SubGroupRole {
      Producer = 0,
      Consumer = 1,
      Store = 2,
      Epilogue = 3,
      NonParticipant
    };

    auto abar_base = allocate_abar_bytes<0, PipelineStorageSize>();
    auto& shared_pipelines = *reinterpret_cast<typename SharedStorage::PipelineStorage*>(abar_base);

    CollectiveMainloop collective_mainloop;
    CollectiveEpilogue collective_epilogue(params.epilogue);

    // Do we load source tensor C or other aux inputs
    bool is_epi_load_needed = false;
    bool is_first_cta_in_cluster = false;
    IsParticipant is_participant = {
      (warp_category == WarpCategory::MMA),                                 // mma
      (warp_category == WarpCategory::Sched) && is_first_cta_in_cluster,    // sched
      (warp_category == WarpCategory::MainloopLoad),                        // main_load
      (warp_category == WarpCategory::EpilogueStore),                       // epi_store
      (warp_category == WarpCategory::Epilogue)                             // epilogue
    };

    // Mainloop Load pipeline
    typename MainloopPipeline::Params mainloop_pipeline_params;
    mainloop_pipeline_params.local_id = local_id;
    MainloopPipeline mainloop_pipeline(shared_pipelines.mainloop, mainloop_pipeline_params);

    // Mainloop-Epilogue pipeline
    typename AccumulatorPipeline::Params accumulator_pipeline_params;
    accumulator_pipeline_params.local_id = local_id;
    AccumulatorPipeline accumulator_pipeline(shared_pipelines.accumulator, accumulator_pipeline_params);

    // Epilogue Store pipeline
    typename EpiStorePipeline::Params epi_store_pipeline_params;
    epi_store_pipeline_params.local_id = local_id;
    epi_store_pipeline_params.num_producers = NumEpilogueThreads;
    EpiStorePipeline epi_store_pipeline(shared_pipelines.epi_store, epi_store_pipeline_params);

    auto mainloop_pipe_producer_state = cutlass::xe4::make_producer_start_state<MainloopPipeline>();
    auto mainloop_pipe_consumer_state = MainloopPipelineState{};

    auto accumulator_pipe_producer_state = cutlass::xe4::make_producer_start_state<AccumulatorPipeline>();
    auto accumulator_pipe_consumer_state = AccumulatorPipelineState{};

    auto store_pipe_producer_state = cutlass::xe4::make_producer_start_state<EpiStorePipeline>();
    auto store_pipe_consumer_state = EpiStorePipelineState{};

    auto wg_k = get<2>(TileShape{});
    uint32_t k_tile_count = (K + wg_k -1) / wg_k;

    using SmemLayoutD = typename CollectiveEpilogue::SmemLayoutD;
    auto tensorD = make_tensor(shared_tensors.epilogue.smem_D.data(), SmemLayoutD{});

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

    auto scheduler = TileScheduler(params.scheduler);
    auto work_tile_info = scheduler.initial_work_tile_info();

    auto coop_ids = params.scheduler.coop_ids;
    auto cluster_masks = params.scheduler.cluster_masks;

    if (is_participant.main_load) {
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
    } else if (is_participant.mma) {
      while (work_tile_info.is_valid()) {
        mainloop_pipe_consumer_state = collective_mainloop.mma(
          cute::make_tuple(mainloop_pipeline, epi_store_pipeline, accumulator_pipeline),
          cute::make_tuple(mainloop_pipe_consumer_state, store_pipe_producer_state, accumulator_pipe_producer_state),
          tensorD, k_tile_count, local_id, cluster_masks, shared_tensors.mainloop);

        // Get next work tile
        ++store_pipe_producer_state;
        ++accumulator_pipe_producer_state;
        scheduler.advance_to_next_work();
        work_tile_info = scheduler.get_current_work();
      } // Scheduler work fetch loop
    } else if (is_participant.epilogue)  {
      uint32_t work_id = local_id - group_info.mainloop_subgroup_num * SubGroupSize;
      while (work_tile_info.is_valid()) {
        collective_epilogue(cute::make_tuple(epi_store_pipeline, accumulator_pipeline),
          cute::make_tuple(store_pipe_producer_state, accumulator_pipe_consumer_state), shared_tensors.epilogue, work_id);

        // Get next work tile
        ++store_pipe_producer_state;
        ++accumulator_pipe_consumer_state;
        scheduler.advance_to_next_work();
        work_tile_info = scheduler.get_current_work();
      } // Scheduler work fetch loop
    } else if (is_participant.epi_store)  {
      while (work_tile_info.is_valid()) {
        auto m_coord = work_tile_info.M_idx;
        auto n_coord = work_tile_info.N_idx;
        auto l_coord = work_tile_info.L_idx;
        auto blk_coord = make_coord(m_coord, n_coord, l_coord);

        collective_epilogue.store(epi_store_pipeline, store_pipe_consumer_state, problem_shape, blk_coord, shared_tensors.epilogue);

        // Get next work tile
        scheduler.advance_to_next_work();
        work_tile_info = scheduler.get_current_work();
      } // Scheduler work fetch loop
    }
  }
};

}
