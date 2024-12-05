#pragma once

#include <sycl/sycl.hpp>
#include <cute/tensor.hpp>

#include "cute/arch/copy_xe4_dma.hpp"
#include "cutlass/epilogue/thread/xe4_conversion_op.hpp"
#include "cutlass/epilogue/thread/xe4_relu_op.hpp"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/epilogue/collective/xe4_epilogue_dma_warpspecialized.hpp"

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
  cute::enable_if_t<cute::is_base_of_v<cutlass::gemm::KernelXe4WarpSpecializedMixedInput, typename CollectiveMainloop_::DispatchPolicy::Schedule>>>
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

    cute::array<ElementAccumulator, cute::cosize_v<SmemLayoutC>> smem_Acc;
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
    auto& problem_shape = params.problem_shape;

    auto ptr = alloc_slm_buffer<uint8_t, SharedStorageSize>(item.get_group());
    auto shared_storage = reinterpret_cast<SharedStorage*>(ptr);

    enum class SubGroupRole {
      Producer0 = 0,
      Producer1,
      Consumer,
      Epilogue,
      Other
    };

    using MainloopPipeline = typename CollectiveMainloop::MainloopPipeline;
    using MainloopPipelineB = typename CollectiveMainloop::MainloopPipelineB;
    using EpilogueStorePipeline = typename CollectiveEpilogue::EpilogueStorePipeline;
    using MainloopPipelineState = typename CollectiveMainloop::PipelineState;
    using MainloopPipelineStateB = typename CollectiveMainloop::PipelineStateB;
    using EpilogueStorePipelineState = typename CollectiveEpilogue::StorePipelineState;

    uint32_t local_id = item.get_local_linear_id();
    MainloopPipeline mainloop_pipeline(local_id);
    MainloopPipelineB mainloop_pipeline_b(local_id);
    EpilogueStorePipeline epilogue_store_pipeline(local_id);

    CollectiveMainloop collective_mainloop;
    CollectiveEpilogue collective_epilogue(params.epilogue);

    MainloopPipelineState mainloop_pipe_consumer_state;
    MainloopPipelineStateB mainloop_pipe_consumer_state_b;
    EpilogueStorePipelineState epilogue_pipe_store_state;

    auto mainloop_pipe_producer_state = cutlass::xe4::make_producer_start_state<MainloopPipeline>();
    auto mainloop_pipe_producer_state_b = cutlass::xe4::make_producer_start_state<MainloopPipelineB>();

    auto K = get<2>(problem_shape);
    auto wg_k = get<2>(TileShape{});
    uint32_t k_tile_count = (K + wg_k -1) / wg_k;

    auto cmLayoutC = upcast<sizeof(ElementAccumulator)>(make_layout(Shape<_32,_32>{}, GenRowMajor{}));
    auto slmLayoutC = tile_to_shape(cmLayoutC, SmemLayoutC{}, Step<_2,_1>{});
    auto accumulator = make_tensor(reinterpret_cast<ElementAccumulator *>(shared_storage->smem_Acc.data()), slmLayoutC);

    auto blk_coord = cute::make_tuple(item.get_group(1), item.get_group(2), 0);
    auto cluster_mask = collective_mainloop.calculateClusterMasks();

    auto warp_group_role = [=]() {
      auto group_info = params.group_info;
      if (local_id == 0) {
        return SubGroupRole::Producer0;
      } else if (local_id == group_info.subgroup_size) {
        return SubGroupRole::Producer1;
      } else if (local_id == 2 * group_info.subgroup_size) {
        return SubGroupRole::Consumer;
      } else if (local_id >= group_info.mainloop_subgroup_num * group_info.subgroup_size) {
        return SubGroupRole::Epilogue;
      } else {
        return SubGroupRole::Other;
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

    if (warp_group_role == SubGroupRole::Producer0 || warp_group_role == SubGroupRole::Producer1) {
      auto [gA_mkl, gB_nkl, gMetaA_mkl, gMetaB_nkl] = collective_mainloop.load_init(problem_shape, params.mainloop);
      if (warp_group_role == SubGroupRole::Producer0) {
        collective_mainloop.loadA(params.mainloop, mainloop_pipeline, mainloop_pipe_producer_state, make_tuple(gA_mkl, gMetaA_mkl), blk_coord, k_tile_count, local_id, cluster_mask, shared_storage->tensors.mainloop);
      } else if (warp_group_role == SubGroupRole::Producer1) {
        collective_mainloop.loadB(params.mainloop, mainloop_pipeline_b, mainloop_pipe_producer_state_b, make_tuple(gB_nkl, gMetaB_nkl), blk_coord, k_tile_count, local_id, cluster_mask, shared_storage->tensors.mainloop);
      }
    } else if (warp_group_role == SubGroupRole::Consumer) {
      collective_mainloop.mma(mainloop_pipeline, mainloop_pipe_consumer_state, mainloop_pipeline_b, mainloop_pipe_consumer_state_b, epilogue_store_pipeline, epilogue_pipe_store_state, accumulator, k_tile_count, local_id, cluster_mask, shared_storage->tensors.mainloop);
    }

    item.barrier(access::fence_space::local_space);

    if (warp_group_role == SubGroupRole::Epilogue) {
      collective_epilogue(accumulator, shared_storage->tensors.epilogue, local_id);
    }

    item.barrier(access::fence_space::local_space);

    if (warp_group_role == SubGroupRole::Consumer) {
      collective_epilogue.store(epilogue_store_pipeline, epilogue_pipe_store_state, problem_shape, blk_coord, shared_storage->tensors.epilogue);
    }
  }
};

}
