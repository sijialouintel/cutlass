#pragma once

#include "cute/arch/copy_xe4_dma.hpp"
#include "cute/atom/copy_traits_xe4_dma.hpp"
#include "cute/container/array.hpp"
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
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
  class SmemLayoutC_,
  class SmemLayoutD_,
  class TileShape_,
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
  using TileShape = TileShape_;

  using TensorDescPtr = uint64_t*;
  using AbarrierPtr = uint64_t*;

  using SmemLayoutC = SmemLayoutC_;
  using GmemTiledCopyC = cute::xe4::ASYNC_TENSOR_LOAD;
  using AuxParamsC = AuxParams<slm_matrix_type::type1, TensorDescPtr, 2>;

  using SmemLayoutD = SmemLayoutD_;
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
    ElementD const* ptr_D = nullptr;
    StrideC dD{};
  };

  // Device side epilogue params
  struct Params
  {
    using TiledStoreD = decltype(make_xe4_copy<GmemTiledCopyD, AuxParamsD>(
      make_tensor(static_cast<ElementD const*>(nullptr), repeat_like(StrideD{}, int32_t(0)), StrideD{}),
      SmemLayoutD{}, make_shape(shape<0>(TileShape{}), shape<1>(TileShape{}))));

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

  template <class ElementType>
  CUTLASS_HOST_DEVICE static auto construct_cm_layout() {
    static constexpr int row_per_core_tile = 2;
    static constexpr int esub_bank_per_cm = 4;
    static constexpr int slm_bank = 4;
    static constexpr int core_tile_size = 64;
    static constexpr int cm_n = 32 / sizeof(ElementType);

    auto cm_row_layout = Layout<Int<cm_n>, _1> {}; // (cm_col_num:_1)
    HOST_PRINT(cm_row_layout);

    auto cm_layout = zipped_product(cm_row_layout,
            Layout<Shape<Int<row_per_core_tile>, Int<esub_bank_per_cm>, Int<slm_bank>>,
                    Stride<_1, Int<row_per_core_tile * esub_bank_per_cm>,
                            Int<row_per_core_tile>>> {});
    HOST_PRINT(cm_layout);

    return cm_layout;
  }

  template <class ElementType, class TileShape, class CoreMatrixLayout>
  CUTLASS_HOST_DEVICE static auto construct_cm_grid_layout(CoreMatrixLayout cm_layout) {
    constexpr int tile_m = size<0>(TileShape {});
    constexpr int tile_n = size<1>(TileShape {});
    constexpr int cm_m = 32;
    constexpr int cm_n = 32 / sizeof(ElementType);

    static_assert(cm_n == size<0>(cm_layout), "Element type to construct core matrix grid is not same as the one in core_matrix!");
    static_assert(tile_m % cm_m == 0, "Tile M is not dividable by total row of core matrix!");
    static_assert(tile_n % cm_n == 0, "Tile N is not dividable by total column core matrix!");
    static_assert((tile_n / cm_n) % 2 == 0, "Odd column in core matrix gird!");

    auto cm_grid_layout
            = flat_product(cm_layout, Layout<Shape<Int<tile_n / cm_n>, Int<tile_m / cm_m>>> {});
    HOST_PRINT(cm_grid_layout);

    /* Group the core matrix grid coordinate dimensions so that it could be indexed by 1-D coordinate. */
    auto grouped_cm_grid_layout = cute::group<rank(CoreMatrixLayout{}), -1>(cm_grid_layout);
    HOST_PRINT(grouped_cm_grid_layout);

    return grouped_cm_grid_layout;
  }

  template <class ElementType, class CoreMatrixGridLayout>
  CUTLASS_HOST_DEVICE static auto swizzle_cm_grid_layout(CoreMatrixGridLayout cm_grid_layout) {
    constexpr int swizzle_B = 1;
    constexpr int swizzle_M = countr_zero(32 / sizeof(ElementType));
    constexpr int swizzle_S = countr_zero(1024 / sizeof(ElementType)) - swizzle_M;

    using Swizzle = Swizzle<1, swizzle_M, swizzle_S>;
    HOST_PRINT(Swizzle{});

    auto swizzled_cm_grid_layout = composition(Swizzle{}, cm_grid_layout);
    HOST_PRINT(swizzled_cm_grid_layout);

    return swizzled_cm_grid_layout;
  }

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
    auto acc_cm_layout = construct_cm_layout<ElementAccumulator>();
    auto d_cm_layout = construct_cm_layout<ElementD>();

    auto acc_cm_grid_layout = construct_cm_grid_layout<ElementAccumulator, TileShape>(acc_cm_layout);
    auto d_cm_grid_layout = construct_cm_grid_layout<ElementD, TileShape>(d_cm_layout);

    auto acc_swizzled_cm_grid_layout = swizzle_cm_grid_layout<ElementAccumulator>(acc_cm_grid_layout);
    auto d_swizzled_cm_grid_layout = swizzle_cm_grid_layout<ElementD>(d_cm_grid_layout);

    auto sAcc = make_tensor(smem_accumulator.data(), acc_swizzled_cm_grid_layout);
    auto sD = make_tensor(reinterpret_cast<ElementD *>(shared_tensors.smem_D.data()), d_swizzled_cm_grid_layout);
    epilogue_op(sAcc, sD, subgroup_num, local_id);
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