#pragma once

#include "cute/arch/copy_xe4_dma.hpp"
#include "cute/atom/copy_traits_xe4_dma.hpp"
#include "cute/container/array.hpp"
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/xe4_detail.hpp"
#include "cutlass/pipeline/xe4_pipeline.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace collective {

/////////////////////////////////////////////////////////////////////////////////////////////////

using namespace cute;
using namespace cute::detail;
using namespace cutlass::epilogue::thread::detail;

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Applies an element wise operation to all elements within the fragment
/// and writes them out to destination storage.
template <
  class StrideD_,
  class SmemLayoutD_,
  class TileShape_,
  class ThreadEpilogueOp_,
  class EpilogueSchedule_
>
class DefaultEpilogue {
public:
  using ThreadEpilogueOp = ThreadEpilogueOp_;
  using ElementOutput = typename ThreadEpilogueOp::ElementOutput;
  using ElementD = ElementOutput;
  using StrideD = StrideD_;
  using TileShape = TileShape_;

  using TensorDescPtr = uint64_t*;
  using AbarrierPtr = uint64_t*;

  using GmemTiledCopyD = cute::xe4::ASYNC_TENSOR_STORE;
  using AuxParamsD = AuxParams<slm_matrix_type::type1, cute::xe4::GMMA::Major::K, TensorDescPtr, 2>;

  using PostOpPipeline = cutlass::xe4::PipelineTmaAsync<1, 1>;
  using PostOpPipelineState = typename PostOpPipeline::PipelineState;

  using StorePipeline = cutlass::xe4::PipelineTmaAsync<1, 2>;
  using StorePipelineState = typename StorePipeline::PipelineState;

  using SmemLayoutD = decltype(tile_to_shape(
    upcast<sizeof(ElementD)>(make_layout(Shape<_32,_32>{}, GenRowMajor{})),
    take<0,2>(TileShape{}), Step<_2,_1>{}));

  static_assert(cute::rank(StrideD{}) == 3, "StrideD must be rank-3: [M, N, L]");

  struct SharedStorage
  {
    struct TensorStorage
    {
      cute::array<ElementD, cute::cosize_v<SmemLayoutD>> smem_D;
    };
  };

  using TensorStorage = typename SharedStorage::TensorStorage;

  // Host side epilogue arguments
  struct Arguments {
    ElementD const* ptr_D = nullptr;
    StrideD dD{};
  };

  // Device side epilogue params
  struct Params
  {
    using TiledStoreD = decltype(make_xe4_copy<GmemTiledCopyD, AuxParamsD>(
      make_tensor(static_cast<ElementD const*>(nullptr), repeat_like(StrideD{}, int32_t(0)), StrideD{}),
      SmemLayoutD{}, take<0,2>(TileShape{})));

    TiledStoreD store_d;
  };

  //
  // Methods
  //

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(
      ProblemShape const& problem_shape,
      Arguments const& args,
      [[maybe_unused]] void* workspace) {

    auto [M, N, K, L] = problem_shape;
    auto D = make_tensor(args.ptr_D, make_layout(make_shape(M, N, L), args.dD));
    auto store_d = make_xe4_copy<GmemTiledCopyD, AuxParamsD>(D, SmemLayoutD {}, make_shape(shape<0>(TileShape {}), shape<1>(TileShape {})));

    return {store_d};
  }

  // Note: SharedStorage is unused for DefaultEpilogue
  CUTLASS_HOST_DEVICE
  DefaultEpilogue(Params const& params_)
      : params(params_), epilogue_op() { }

  CUTLASS_DEVICE void
  operator()(
      StorePipeline store_pipeline,
      StorePipelineState store_pipe_state,
      PostOpPipeline postop_pipeline,
      PostOpPipelineState& postop_pipe_state,
      TensorStorage& shared_tensors,
      uint32_t local_id)
  {
    store_pipeline.producer_try_wait(store_pipe_state);
    postop_pipeline.consumer_try_wait(postop_pipe_state);

    constexpr auto tile_mn = take<0,2>(TileShape{});
    auto tensor_d = make_tensor(shared_tensors.smem_D.data(), CoreMatrix::retile<ElementD>(tile_mn));
    epilogue_op(tensor_d, tensor_d, local_id);

    store_pipeline.producer_arrive(store_pipe_state, 1);
    ++store_pipe_state;
    ++postop_pipe_state;
  }

  template<
    class ProblemShape,
    class BlockCoordMNL
  >
  CUTLASS_DEVICE void
  store(
      StorePipeline store_pipeline,
      StorePipelineState store_pipe_state,
      PostOpPipeline postop_pipeline,
      PostOpPipelineState postop_pipe_state,
      ProblemShape const& problem_shape,
      BlockCoordMNL blk_coord_mnl,
      TensorStorage& shared_tensors)
  {
    store_pipeline.consumer_try_wait(store_pipe_state);       // Wait for all postop threads finish their calculation
    postop_pipeline.consumer_arrive(postop_pipe_state, 1);       // Notify the mma thread. It can now overwrite the accumulator

    auto sD = make_tensor(shared_tensors.smem_D.data(), SmemLayoutD {});
    auto [M, N, K, L] = problem_shape;
    auto mD_mnl = params.store_d.get_tma_tensor(make_shape(M, N, L)); // (m,n,l)
    auto gD_mnl = flat_divide(mD_mnl, make_shape(shape<0>(TileShape {}), shape<1>(TileShape {}))); // (BLK_M,BLK_N,m,n,l)
    auto block_store_d = params.store_d.get_slice(0);
    auto [m_coord, n_coord, l_coord] = blk_coord_mnl;

    auto gD = gD_mnl(_, _, m_coord, n_coord, l_coord);  // (BLK_M,BLK_N)
    auto tDgD = block_store_d.partition_S(gD);           // (TMA,TMA_M,TMA_N)
    auto tDsD = block_store_d.partition_D(sD);    // (TMA,TMA_M,TMA_N)

    auto abar_store = store_pipeline.consumer_get_barrier(store_pipe_state);
    constexpr uint32_t slm_bytes_store = sizeof(ElementD) * size(SmemLayoutD {});
    copy(params.store_d.with(abar_store), tDsD, tDgD);
    store_pipeline.consumer_commit(store_pipe_state, slm_bytes_store);
    store_pipeline.producer_try_wait(store_pipe_state);
    ++store_pipe_state;
  }

private:
  Params params;
  ThreadEpilogueOp epilogue_op;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace collective
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////