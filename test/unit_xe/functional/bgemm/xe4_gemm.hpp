#pragma once

#include <sycl/sycl.hpp>
#include <cute/tensor.hpp>

#include "collective_mma.hpp"
#include "xe4_copy_async.hpp"

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
  using TileShape = typename CollectiveMainloop::TileShape;
  using TiledMma  = typename CollectiveMainloop::TiledMma;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;

  // Device side arguments
  struct Arguments {
    sycl::nd_item<3> item;
    ProblemShape problem_shape;
    MainloopArguments mainloop;
  };

  // Kernel entry point API
  struct Params {
    sycl::nd_item<3> item;
    ProblemShape problem_shape;
    MainloopParams mainloop;
  };

  static
  Params
  to_underlying_arguments(Arguments const& args, void* workspace) {
    (void) workspace;
    return {
      args.item,
      args.problem_shape,
      CollectiveMainloop::to_underlying_arguments(args.problem_shape, args.mainloop)
    };
  }

  CUTLASS_DEVICE
  void
  operator()(Params const& params) {
    auto& item = params.item;

    using MainloopPipeline = typename CollectiveMainloop::MainloopPipeline;
    MainloopPipeline mainloop_pipeline(item);

    auto problem_shape_MNKL = append<4>(params.problem_shape, Int<1>{});
    auto [M, N, K, L] = problem_shape_MNKL;

    auto tile_shape = TileShape{};
    auto wg_k = get<2>(tile_shape);

    CollectiveMainloop collective_mainloop;
    auto load_inputs = collective_mainloop.load_init(problem_shape_MNKL, params.mainloop);

    typename CollectiveMainloop::PipelineState mainloop_pipe_consumer_state;
    auto mainloop_pipe_producer_state = cutlass::xe4::make_producer_start_state<MainloopPipeline>();

    uint32_t local_id = item.get_local_id(2);
    uint32_t k_tile_count = (K + wg_k -1) / wg_k;
    auto blk_coord = cute::make_tuple(item.get_group(1), item.get_group(2), 0);
    collective_mainloop.load(params.mainloop, mainloop_pipeline, mainloop_pipe_producer_state, load_inputs, blk_coord, k_tile_count, local_id);

    TiledMma tiled_mma;
    auto thread_mma = tiled_mma.get_thread_slice(0);
    auto accumulators = thread_mma.partition_fragment_C(params.mainloop.slm_acc);    // (MMA,MMA_M,MMA_N)

    collective_mainloop.mma(params.mainloop, mainloop_pipeline, mainloop_pipe_consumer_state, accumulators, k_tile_count, local_id);
  }
};

}
