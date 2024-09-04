#pragma once

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cute/container/array.hpp"

#include "cutlass/pipeline/xe4_pipeline.hpp"

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
  class StrideC_,
  class StrideD_,
  class ThreadEpilogueOp_,
  class EpilogueSchedule_
>
class DefaultEpilogue {
public:
  using ThreadEpilogueOp = ThreadEpilogueOp_;
  using ElementOutput = typename ThreadEpilogueOp::ElementOutput;
  using ElementAccumulator = typename ThreadEpilogueOp::ElementAccumulator;
  using ElementC = ElementAccumulator;
  using ElementD = ElementOutput;
  using StrideC = StrideC_;
  using StrideD = StrideD_;
  using TileShape = typename ThreadEpilogueOp::TileShape;

  using TensorDescPtr = uint64_t*;
  using AbarrierPtr = uint64_t*;

  using SmemLayoutC = typename ThreadEpilogueOp::SmemLayoutOutput;
  using GmemTiledCopyC = cute::xe4::ASYNC_TENSOR_LOAD;
  using AuxParamsC = AuxParams<slm_matrix_type::type1, TensorDescPtr, 2>;

  using SmemLayoutD = typename ThreadEpilogueOp::SmemLayoutOutput;
  using GmemTiledCopyD = cute::xe4::ASYNC_TENSOR_STORE;
  using AuxParamsD = AuxParams<slm_matrix_type::type1, TensorDescPtr, 3>;

  using EpilogueLoadPipeline = cutlass::xe4::PipelineTmaAsync<1, AbarrierPtr>;
  using LoadPipelineState = typename EpilogueLoadPipeline::PipelineState;

  using EpilogueStorePipeline = cutlass::xe4::PipelineTmaStore<1, AbarrierPtr>;
  using StorePipelineState = typename EpilogueStorePipeline::PipelineState;

  static_assert(cute::rank(StrideC{}) == 3, "StrideC must be rank-3: [M, N, L]");

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
    ElementC const* ptr_C = nullptr;
    StrideC dC{};
    ElementD const* ptr_D = nullptr;
    StrideC dD{};
  };

  // Device side epilogue params
  struct Params
  {
    using TiledLoadC = decltype(make_xe4_copy<GmemTiledCopyC, AuxParamsC>(
      make_tensor(static_cast<ElementC const*>(nullptr), repeat_like(StrideC{}, int32_t(0)), StrideC{}),
      SmemLayoutC{}, make_shape(shape<0>(TileShape{}), shape<1>(TileShape{}))));

    using TiledStoreD = decltype(make_xe4_copy<GmemTiledCopyD, AuxParamsD>(
      make_tensor(static_cast<ElementD const*>(nullptr), repeat_like(StrideD{}, int32_t(0)), StrideD{}),
      SmemLayoutD{}, make_shape(shape<0>(TileShape{}), shape<1>(TileShape{}))));

    TiledLoadC load_c;
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

    auto C = make_tensor(args.ptr_C, make_layout(make_shape(M, N, L), args.dC));
    auto load_c = make_xe4_copy<GmemTiledCopyC, AuxParamsC>(C, SmemLayoutC {}, make_shape(shape<0>(TileShape {}), shape<1>(TileShape {})));

    auto D = make_tensor(args.ptr_D, make_layout(make_shape(M, N, L), args.dD));
    auto store_d = make_xe4_copy<GmemTiledCopyD, AuxParamsD>(D, SmemLayoutD {}, make_shape(shape<0>(TileShape {}), shape<1>(TileShape {})));

    return {load_c, store_d};
  }

  // Note: SharedStorage is unused for DefaultEpilogue
  CUTLASS_HOST_DEVICE
  DefaultEpilogue(Params const& params_)
      : params(params_), epilogue_op() { }

  template<
    class TensorAccumulator
  >
  CUTLASS_DEVICE void
  operator()(
      TensorAccumulator smem_accumulator,
      TensorStorage& shared_tensors,
      uint32_t subgroup_num,
      uint32_t local_id)
  {
    auto sD = make_tensor(reinterpret_cast<ElementD *>(shared_tensors.smem_D.data()), SmemLayoutD {});

    epilogue_op(smem_accumulator, sD, subgroup_num, local_id);
  }

  template<
    class ProblemShape,
    class BlockCoordMNL
  >
  CUTLASS_DEVICE void
  store(
      EpilogueStorePipeline epilogue_store_pipeline,
      StorePipelineState pipe_store_state,
      ProblemShape const& problem_shape,
      BlockCoordMNL blk_coord_mnl,
      TensorStorage& shared_tensors)
  {
    auto sD = make_tensor(reinterpret_cast<ElementD *>(shared_tensors.smem_D.data()), SmemLayoutD {});
    auto [M, N, K, L] = problem_shape;
    auto mD_mnl = params.store_d.get_tma_tensor(make_shape(M, N, L)); // (m,n,l)
    auto gD_mnl = flat_divide(mD_mnl, make_shape(shape<0>(TileShape {}), shape<1>(TileShape {}))); // (BLK_M,BLK_N,m,n,l)
    auto block_store_d = params.store_d.get_slice(0);
    auto [m_coord, n_coord, l_coord] = blk_coord_mnl;

    auto gD = gD_mnl(_, _, m_coord, n_coord, l_coord);  // (BLK_M,BLK_N)
    auto tDgD = block_store_d.partition_S(gD);           // (TMA,TMA_M,TMA_N)
    auto tDsD = block_store_d.partition_D(sD);    // (TMA,TMA_M,TMA_N)

    auto abar_store = epilogue_store_pipeline.store_get_barrier(pipe_store_state);
    constexpr uint32_t slm_bytes_store = sizeof(ElementD) * size(SmemLayoutD {});
    copy(params.store_d.with(abar_store), tDsD, tDgD);
    epilogue_store_pipeline.store_commit(pipe_store_state, slm_bytes_store);
    epilogue_store_pipeline.store_try_wait(pipe_store_state);
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