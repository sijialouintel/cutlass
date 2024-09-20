#pragma once

#include <sycl/sycl.hpp>
#include <cute/tensor.hpp>

#include "cute/arch/copy_xe4_dma.hpp"
#include "cutlass/epilogue/thread/xe4_conversion_op.hpp"
#include "cutlass/epilogue/thread/xe4_relu_op.hpp"
#include "cutlass/gemm/collective/xe4_mma_dma_amma_ss_warpspecialized.hpp"
#include "cutlass/epilogue/collective/xe4_epilogue_dma_warpspecialized.hpp"

namespace cutlass::gemm::kernel {

using namespace cute;
using namespace sycl;
using namespace cute::xe4;

template <
  class ProblemShapeOrThreadblockMma_, // (m, n, k) or (m, n, k, l)
  class CollectiveMainloopOrEpilogue_,
  class CollectiveEpilogueOrThreadblockSwizzle_,
  class TileScheduler_ = void,
  class Enable = void
>
class GemmUniversal;

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
  TileScheduler_>
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
  using ClusterShape = typename DispatchPolicy::ClusterShape;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;
  using SmemLayoutC = typename CollectiveEpilogue::SmemLayoutC;

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

  // Device side arguments
  struct Arguments {
    sycl::nd_item<3> item;
    ProblemShape problem_shape;
    MainloopArguments mainloop;
    EpilogueArguments epilogue;
  };

  // Kernel entry point API
  struct Params {
    sycl::nd_item<3> item;
    ProblemShape problem_shape;
    MainloopParams mainloop;
    EpilogueParams epilogue;
    SharedStorage* shared_storage = nullptr;
  };

  static
  Params
  to_underlying_arguments(Arguments const& args, void* workspace) {
    (void) workspace;

    auto ptr = sycl::ext::oneapi::group_local_memory_for_overwrite<uint8_t[SharedStorageSize]>(args.item.get_group());

    return {
      args.item,
      args.problem_shape,
      CollectiveMainloop::to_underlying_arguments(args.problem_shape, args.mainloop),
      CollectiveEpilogue::to_underlying_arguments(args.problem_shape, args.epilogue, nullptr),
      reinterpret_cast<SharedStorage*>(*ptr)
    };
  }

  CUTLASS_DEVICE
  void
  operator()(Params const& params) {
    auto& item = params.item;
    auto& problem_shape = params.problem_shape;
    auto shared_storage = params.shared_storage;

    enum class SubGroupRole {
      Producer = 0,
      Consumer = 1,
      Epilogue = 2,
      Other
    };

    using MainloopPipeline = typename CollectiveMainloop::MainloopPipeline;
    using EpilogueStorePipeline = typename CollectiveEpilogue::EpilogueStorePipeline;
    using MainloopPipelineState = typename CollectiveMainloop::PipelineState;
    using EpilogueStorePipelineState = typename CollectiveEpilogue::StorePipelineState;

    uint32_t local_id = item.get_local_linear_id();
    MainloopPipeline mainloop_pipeline(local_id);
    EpilogueStorePipeline epilogue_store_pipeline(local_id);

    CollectiveMainloop collective_mainloop;
    CollectiveEpilogue collective_epilogue(params.epilogue);

    MainloopPipelineState mainloop_pipe_consumer_state;
    EpilogueStorePipelineState epilogue_pipe_store_state;
    auto mainloop_pipe_producer_state = cutlass::xe4::make_producer_start_state<MainloopPipeline>();

    auto K = get<2>(problem_shape);
    auto wg_k = get<2>(TileShape{});
    uint32_t k_tile_count = (K + wg_k -1) / wg_k;

    auto cmLayoutC = upcast<sizeof(ElementAccumulator)>(make_layout(Shape<_32,_32>{}, GenRowMajor{}));
    auto slmLayoutC = tile_to_shape(cmLayoutC, SmemLayoutC{}, Step<_2,_1>{});
    auto accumulator = make_tensor(reinterpret_cast<ElementAccumulator *>(shared_storage->smem_Acc.data()), slmLayoutC);

    auto blk_coord = cute::make_tuple(item.get_group(1), item.get_group(2), 0);
    auto cluster_mask = collective_mainloop.calculateClusterMasks();

    auto warp_group_role = [=]() {
      if (local_id == 0) {
        return SubGroupRole::Producer;
      } else if (local_id == 32) {
        return SubGroupRole::Consumer;
      } else if (local_id >= 128) {
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

    if (warp_group_role == SubGroupRole::Producer) {
      auto load_inputs = collective_mainloop.load_init(problem_shape, params.mainloop);
      static_assert(cute::tuple_size_v<decltype(load_inputs)> >= 2, "Output of load_init must have at least two elements (A, B)");
      collective_mainloop.load(params.mainloop, mainloop_pipeline, mainloop_pipe_producer_state, load_inputs, blk_coord, k_tile_count, local_id, cluster_mask, shared_storage->tensors.mainloop);
    } else if (warp_group_role == SubGroupRole::Consumer) {
      collective_mainloop.mma(params.mainloop, mainloop_pipeline, mainloop_pipe_consumer_state, accumulator, k_tile_count, local_id, cluster_mask, shared_storage->tensors.mainloop);
    }

    item.barrier(access::fence_space::local_space);

    if (warp_group_role == SubGroupRole::Epilogue) {
      collective_epilogue(accumulator, shared_storage->tensors.epilogue, 4, local_id);
    }

    item.barrier(access::fence_space::local_space);

    if (warp_group_role == SubGroupRole::Consumer) {
      collective_epilogue.store(epilogue_store_pipeline, epilogue_pipe_store_state, problem_shape, blk_coord, shared_storage->tensors.epilogue);
    }
  }
};

}
