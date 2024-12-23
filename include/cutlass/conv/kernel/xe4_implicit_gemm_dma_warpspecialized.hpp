#pragma once

#include <sycl/sycl.hpp>
#include <cute/tensor.hpp>
#include "cute/arch/copy_xe4_dma.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/epilogue/collective/xe4_epilogue_dma_warpspecialized_implicit_gemm.hpp"

namespace cutlass::conv::kernel {

using namespace cute;
using namespace sycl;
using namespace cute::xe4;
// using namespace cutlass::gemm::kernel;

template <
  class ProblemShape_,
  class CollectiveMainloop_,
  class CollectiveEpilogue_,
  class TileScheduler_
>
class Xe4ConvUniversal
{
public:
  //
  // Type Aliases
  //
  using ProblemShape = ProblemShape_;

  // Handles the static_assert placed inside the operator()
  // This is also used to decide whether the load_init inside collective mainloop returns rank 4 tensors or rank 5 tensors
  static constexpr bool IsConvProblemShape = not (cute::is_tuple_v<ProblemShape>|| cutlass::gemm::kernel::IsCutlass3ArrayKernel<ProblemShape>::value);
  static_assert( IsConvProblemShape || (cute::rank(ProblemShape{}) == 3 || cute::rank(ProblemShape{}) == 4), "ProblemShape{} should be <M,N,K> or <M,N,K,L> for Gemm");

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
  using ElementD = typename CollectiveEpilogue::ElementD;

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
    ProblemShape problem_shape;
    MainloopArguments mainloop;
    EpilogueArguments epilogue;
  };

  // Kernel entry point API
  struct Params {
    ProblemShape problem_shape;
    MainloopParams mainloop;
    EpilogueParams epilogue;
  };

  static Params
  to_underlying_arguments(Arguments const& args, void* workspace) {
    (void) workspace;

    return {
      args.problem_shape,
      CollectiveMainloop::to_underlying_arguments(args.problem_shape, args.mainloop),
      CollectiveEpilogue::to_underlying_arguments(args.problem_shape, args.epilogue)
    };
  }

  CUTLASS_DEVICE
  void
  operator()(Params const& params, sycl::nd_item<3> item) {
    auto& problem_shape = params.problem_shape;

    auto ptr = alloc_slm_buffer<uint8_t, SharedStorageSize>(item.get_group());
    auto shared_storage = reinterpret_cast<SharedStorage*>(ptr);

    enum class SubGroupRole {
      Producer = 0,
      Consumer = 1,
      EpiloguePostOp = 2,
      Other
    };

    using MainloopPipeline = typename CollectiveMainloop::MainloopPipeline;
    using EpilogueStorePipeline = typename CollectiveEpilogue::EpilogueStorePipeline;
    using MainloopPipelineState = typename CollectiveMainloop::PipelineState;
    using EpilogueStorePipelineState = typename CollectiveEpilogue::StorePipelineState;

    uint32_t local_id = item.get_local_linear_id();
    constexpr uint32_t abar_count = 2 * (MainloopPipeline::Stages + EpilogueStorePipeline::Stages);

    auto abar_base = allocate_abar<0, abar_count>();
    MainloopPipeline mainloop_pipeline(abar_base, local_id);
    auto abar_store_base = abar_base + 2 * MainloopPipeline::Stages;
    EpilogueStorePipeline epilogue_store_pipeline(abar_store_base, local_id);

    CollectiveMainloop collective_mainloop;
    CollectiveEpilogue collective_epilogue;

    MainloopPipelineState mainloop_pipe_consumer_state;
    EpilogueStorePipelineState epilogue_pipe_store_consumer_state;
    auto mainloop_pipe_producer_state = cutlass::xe4::make_producer_start_state<MainloopPipeline>();
    auto epilogue_pipe_store_producer_state = cutlass::xe4::make_producer_start_state<EpilogueStorePipeline>();

    using SmemLayoutAtomAcc = decltype(make_layout(Shape<_32, Int<32 / sizeof(ElementAccumulator)>>{}, GenRowMajor{}));
    using SmemLayoutAtomDst = decltype(make_layout(Shape<_32, Int<32 / sizeof(ElementD)>>{}, GenRowMajor{}));

    using SmemLayoutAcc = decltype(tile_to_shape(
        SmemLayoutAtomAcc{},
        make_shape(shape<0>(TileShape{}), shape<1>(TileShape{})),
        Step<_2,_1>{}));
    using SmemLayoutDst = decltype(tile_to_shape(
        SmemLayoutAtomDst{},
        make_shape(shape<0>(TileShape{}), shape<1>(TileShape{})),
        Step<_2,_1>{}));

    auto accumulator = make_tensor(reinterpret_cast<ElementAccumulator*>(shared_storage->smem_Acc.data()), SmemLayoutAcc {});
    auto dst = make_tensor(reinterpret_cast<ElementD*>(shared_storage->tensors.epilogue.smem_D.data()), SmemLayoutDst {});

    auto conv_problem_shape = collective_mainloop.get_problem_shape_MNKL(problem_shape);

    auto load_inputs = collective_mainloop.load_init(conv_problem_shape, params.mainloop);
    auto [gA_mk, gB_nk] = load_inputs;
    auto k_tile_iter = cute::make_coord_iterator(shape<3>(gA_mk));
    auto k_tile_count = size<3>(gA_mk);

    uint32_t subgroup_id = local_id / 32;
    auto warp_group_role = [=]() {
      if (subgroup_id == 0) {
        return SubGroupRole::Producer;
      } else if (local_id == 32) {
        return SubGroupRole::Consumer;
      } else if (subgroup_id == 2) {
        return SubGroupRole::EpiloguePostOp;
      } else {
        return SubGroupRole::Other;
      }
    } ();

    item.barrier(access::fence_space::local_space);

    if(warp_group_role == SubGroupRole::Producer){
      auto n_coord = idx2crd(int(item.get_group(2)), shape<2>(gB_nk), compact_col_major(shape<2>(gB_nk)));
      auto blk_coord = make_tuple(uint32_t(item.get_group(1)), n_coord, 0, 0);

      collective_mainloop.load(params.mainloop, mainloop_pipeline, mainloop_pipe_producer_state, load_inputs, blk_coord, k_tile_iter, k_tile_count, local_id, shared_storage->tensors.mainloop);
    }
    else if (warp_group_role == SubGroupRole::Consumer){
      collective_mainloop.mma(mainloop_pipeline, mainloop_pipe_consumer_state, epilogue_store_pipeline, epilogue_pipe_store_producer_state, accumulator, dst, k_tile_count, local_id, ptr);
    }

    if (warp_group_role == SubGroupRole::EpiloguePostOp) {
      collective_epilogue.store(params.epilogue, epilogue_store_pipeline, epilogue_pipe_store_consumer_state, conv_problem_shape, local_id - 64, shared_storage->tensors.epilogue);
    }
  }
};

}
