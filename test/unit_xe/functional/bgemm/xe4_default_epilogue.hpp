#pragma once

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cute/container/array.hpp"

#include "pipe.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace collective {

/////////////////////////////////////////////////////////////////////////////////////////////////

using namespace cute;
using namespace cute::detail;

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Applies an element wise operation to all elements within the fragment
/// and writes them out to destination storage.
template <
  int StagesC_,
  class StrideC_,
  class ThreadEpilogueOp_,
  class EpilogueSchedule_
>
class DefaultEpilogue {
public:
  using ThreadEpilogueOp = ThreadEpilogueOp_;
  using ElementOutput = typename ThreadEpilogueOp::ElementOutput;
  using ElementAccumulator = typename ThreadEpilogueOp::ElementAccumulator;
  using ElementC = ElementOutput;
  using StrideC = StrideC_;
  using TileShape = typename ThreadEpilogueOp::TileShape;

  constexpr static int StagesC = StagesC_;

  using TensorDescPtr = uint64_t*;
  using AbarrierPtr = uint64_t*;

  using SmemLayoutC = typename ThreadEpilogueOp::SmemLayoutOutput;
  using GmemTiledCopyC = cute::xe4::ASYNC_TENSOR_STORE;

  using AuxParamsC = AuxParams<cm_size_t::cm_32x32B, cm_layout_t::vertical_split, false, TensorDescPtr, 2>;

  using EpiloguePipeline = cutlass::xe4::PipelineTmaStore<StagesC, AbarrierPtr>;
  using PipelineState = typename EpiloguePipeline::PipelineState;

  static_assert(cute::rank(StrideC{}) == 3, "StrideC must be rank-3: [M, N, L]");

  struct SharedStorage
  {
    struct TensorStorage
    {
      cute::array<ElementC, cute::cosize_v<SmemLayoutC>> smem_C;
    };
  };

  using TensorStorage = typename SharedStorage::TensorStorage;

  // Host side epilogue arguments
  struct Arguments {
    ElementC const* ptr_C = nullptr;
    StrideC dC{};
  };

  // Device side epilogue params
  struct Params
  {
    using TiledStoreC = decltype(make_xe4_copy<GmemTiledCopyC, AuxParamsC>(
      make_tensor(static_cast<ElementC const*>(nullptr), repeat_like(StrideC{}, int32_t(0)), StrideC{}),
      SmemLayoutC{}, make_shape(shape<0>(TileShape{}), shape<1>(TileShape{}))));

    TiledStoreC store_c;
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
    auto C = make_tensor(args.ptr_C, make_layout(make_shape(M, N, L), args.dC));
    auto store_c = make_xe4_copy<GmemTiledCopyC, AuxParamsC>(C, SmemLayoutC{}, make_shape(shape<0>(TileShape{}), shape<1>(TileShape{})));

    return {store_c};
  }

  // Note: SharedStorage is unused for DefaultEpilogue
  CUTLASS_HOST_DEVICE
  DefaultEpilogue(Params const& params_)
      : params(params_), epilogue_op() { }

  template<
    class ProblemShape,
    class BlockCoordMNL,
    class TensorAccumulator
  >
  CUTLASS_DEVICE void
  operator()(
      EpiloguePipeline epilogue_pipeline,
      PipelineState pipe_store_state,
      ProblemShape const& problem_shape,
      BlockCoordMNL blk_coord_mnl,
      TensorAccumulator smem_accumulator,
      uint32_t local_id,
      TensorStorage& shared_tensors)
  {
    auto sC = make_tensor(reinterpret_cast<ElementC *>(shared_tensors.smem_C.data()), SmemLayoutC {});

    epilogue_op(smem_accumulator, sC);

    if (local_id == 0) {
      auto [M, N, K, L] = problem_shape;
      auto mC_mnl = params.store_c.get_tma_tensor(make_shape(M, N, L)); // (m,n,l)
      auto gC_mnl = flat_divide(mC_mnl, make_shape(shape<0>(TileShape {}), shape<1>(TileShape {}))); // (BLK_M,BLK_N,m,n,l)
      auto block_load_c = params.store_c.get_slice(0);
      auto [m_coord, n_coord, l_coord] = blk_coord_mnl;

      auto gC = gC_mnl(_, _, m_coord, n_coord, l_coord);  // (BLK_M,BLK_N)
      auto tCgC = block_load_c.partition_S(gC);           // (TMA,TMA_M,TMA_N)
      auto tCsC = block_load_c.partition_D(smem_accumulator);    // (TMA,TMA_M,TMA_N)

      auto abar_store = epilogue_pipeline.store_get_barrier(pipe_store_state);
      constexpr uint32_t slm_bytes_store = sizeof(ElementC) * size(SmemLayoutC {});
      copy(params.store_c.with(abar_store), tCsC, tCgC);
      epilogue_pipeline.store_commit(pipe_store_state, slm_bytes_store);
      epilogue_pipeline.store_try_wait(pipe_store_state);
    }
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