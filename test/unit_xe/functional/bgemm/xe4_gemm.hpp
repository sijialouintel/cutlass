#pragma once

#include <sycl/sycl.hpp>
#include <cute/tensor.hpp>

#include "collective_mma.hpp"
#include "xe4_copy_async.hpp"

#include "conversion_op.hpp"
#include "xe4_default_epilogue.hpp"

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
  using ProblemShape = ProblemShape_;
  using CollectiveMainloop = CollectiveMainloop_;
  using CollectiveEpilogue = CollectiveEpilogue_;
  using TileShape = typename CollectiveMainloop::TileShape;
  using TiledMma  = typename CollectiveMainloop::TiledMma;
  using ElementA  = typename CollectiveMainloop::ElementA;
  using StrideA   = typename CollectiveMainloop::StrideA;
  using SmemLayoutA = typename CollectiveMainloop::SmemLayoutA;
  using ElementB  = typename CollectiveMainloop::ElementB;
  using StrideB   = typename CollectiveMainloop::StrideB;
  using SmemLayoutB = typename CollectiveMainloop::SmemLayoutB;
  using ElementAccumulator = typename CollectiveMainloop::ElementAccumulator;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;

  using ElementC  = typename CollectiveEpilogue::ElementC;
  using SmemLayoutC = typename CollectiveEpilogue::SmemLayoutC;

  using SlmTensorAcc = decltype(make_tensor(static_cast<ElementAccumulator*>(nullptr), SmemLayoutC{}));

  struct SharedStorage
  {
    using MainloopTensorStorage = typename CollectiveMainloop::TensorStorage;
    using EpilogueTensorStorage = typename CollectiveEpilogue::TensorStorage;

    struct TensorStorage
    {
      MainloopTensorStorage mainloop;
      EpilogueTensorStorage epilogue;
    } tensors;

    cute::array<ElementAccumulator, cute::cosize_v<SmemLayoutC>> accumulator;
  };

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

    auto ptr = sycl::ext::oneapi::group_local_memory_for_overwrite<uint8_t[sizeof(SharedStorage)]>(args.item.get_group());

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

    using MainloopPipeline = typename CollectiveMainloop::MainloopPipeline;
    MainloopPipeline mainloop_pipeline(item);
    using EpiloguePipeline = typename CollectiveEpilogue::EpiloguePipeline;
    EpiloguePipeline epilogue_pipeline(item);

    auto problem_shape_MNKL = append<4>(params.problem_shape, Int<1>{});
    auto [M, N, K, L] = problem_shape_MNKL;

    auto tile_shape = TileShape{};
    auto wg_k = get<2>(tile_shape);

    CollectiveMainloop collective_mainloop;
    auto load_inputs = collective_mainloop.load_init(problem_shape_MNKL, params.mainloop);
    CollectiveEpilogue collective_epilogue(params.epilogue);

    typename CollectiveMainloop::PipelineState mainloop_pipe_consumer_state;
    auto mainloop_pipe_producer_state = cutlass::xe4::make_producer_start_state<MainloopPipeline>();
    typename CollectiveEpilogue::PipelineState epilogue_pipe_store_state;

    uint32_t local_id = item.get_local_linear_id();
    uint32_t k_tile_count = (K + wg_k -1) / wg_k;

    auto accumulator = make_tensor(reinterpret_cast<ElementAccumulator *>(params.shared_storage->accumulator.data()), SmemLayoutC {});
    auto blk_coord = cute::make_tuple(item.get_group(1), item.get_group(2), 0);
    auto cluster_mask = collective_mainloop.calculateClusterMasks();

    if (local_id == 0) {
      collective_mainloop.load(params.mainloop, mainloop_pipeline, mainloop_pipe_producer_state, load_inputs, blk_coord, k_tile_count, local_id, cluster_mask, params.shared_storage->tensors.mainloop);
    } else if (local_id == 32) {
      collective_mainloop.mma(params.mainloop, mainloop_pipeline, mainloop_pipe_consumer_state, accumulator, k_tile_count, local_id, cluster_mask, params.shared_storage->tensors.mainloop);
    }

    item.barrier(access::fence_space::local_space);

    collective_epilogue(epilogue_pipeline, epilogue_pipe_store_state, problem_shape_MNKL, blk_coord, accumulator, local_id, params.shared_storage->tensors.epilogue);
  }
};

}
