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
  };

  static
  Params
  to_underlying_arguments(Arguments const& args, void* workspace) {
    (void) workspace;

    return {
      args.group_info,
      args.problem_shape,
      CollectiveMainloop::to_underlying_arguments(args.problem_shape, args.mainloop),
      CollectiveEpilogue::to_underlying_arguments(args.problem_shape, args.epilogue, nullptr)
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
    MainloopPipeline mainloop_pipeline(local_id);
    EpiloguePipeline epilogue_pipeline(local_id);
    StorePipeline epilogue_store_pipeline(local_id, group_info.epilogue_subgroup_num * group_info.subgroup_size, 1);

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

    auto blk_coord = cute::make_tuple(item.get_group(1), item.get_group(2), 0);
    auto cluster_mask = collective_mainloop.calculateClusterMasks();

    auto warp_group_role = [=]() {
      if (local_id == 0) {
        return SubGroupRole::Producer;
      } else if (local_id == group_info.subgroup_size) {
        return SubGroupRole::Consumer;
      } else if (local_id == 2 * group_info.subgroup_size) {
        return SubGroupRole::Store;
      } else if (local_id >= group_info.mainloop_subgroup_num * group_info.subgroup_size) {
        return SubGroupRole::Epilogue;
      } else {
        return SubGroupRole::NonParticipant;
      }
    } ();

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

    if (warp_group_role == SubGroupRole::Producer) {
      auto load_inputs = collective_mainloop.load_init(problem_shape, params.mainloop);
      static_assert(cute::tuple_size_v<decltype(load_inputs)> >= 2, "Output of load_init must have at least two elements (A, B)");
      collective_mainloop.load(params.mainloop, mainloop_pipeline, mainloop_pipe_producer_state, load_inputs, blk_coord, k_tile_count, local_id, cluster_mask, shared_tensors.mainloop);
    } else if (warp_group_role == SubGroupRole::Consumer) {
      collective_mainloop.mma(mainloop_pipeline, mainloop_pipe_consumer_state, epilogue_pipeline, epilogue_pipe_producer_state, tensorD, k_tile_count, local_id, cluster_mask, shared_tensors.mainloop);
    } else if (warp_group_role == SubGroupRole::Epilogue) {
      uint32_t work_id = local_id - group_info.mainloop_subgroup_num * group_info.subgroup_size;
      collective_epilogue(epilogue_store_pipeline, store_pipe_producer_state, epilogue_pipeline, epilogue_pipe_consumer_state, shared_tensors.epilogue, work_id);
    } else if (warp_group_role == SubGroupRole::Store) {
      collective_epilogue.store(epilogue_store_pipeline, store_pipe_consumer_state, epilogue_pipeline, epilogue_pipe_consumer_state, problem_shape, blk_coord, shared_tensors.epilogue);
    }
  }
};

}
