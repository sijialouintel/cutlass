#pragma once

#include "cute/arch/copy_xe4_dma.hpp"
#include "cute/atom/copy_traits_xe4_dma.hpp"
#include "cute/container/array.hpp"
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/xe4_detail.hpp"
#include "cutlass/pipeline/xe4_pipeline.hpp"
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace collective {

/////////////////////////////////////////////////////////////////////////////////////////////////

using namespace cute;
using namespace cute::detail;
using namespace cutlass::epilogue;
using namespace cutlass::epilogue::thread::detail;

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Applies an element wise operation to all elements within the fragment
/// and writes them out to destination storage.
template <
  int FragmentSize,
  uint32_t EpiSgNum,
  uint32_t SgSize,
  class ElementD_,
  class StrideD_,
  class SmemLayoutD_,
  class TileShape_,
  class FusionCallbacks_
>
class CollectiveEpilogue<
  Xe4DmaWarpSpecialized<FragmentSize, EpiSgNum, SgSize>,
  ElementD_,
  StrideD_,
  SmemLayoutD_,
  TileShape_,
  FusionCallbacks_
> {
public:
  using ElementD = ElementD_;
  using StrideD = StrideD_;
  using TileShape = TileShape_;
  using FusionCallbacks = FusionCallbacks_;

  using TensorDesc = uint64_t*;
  using TiledCopyD = cute::xe4::ASYNC_TENSOR_STORE;
  using AuxParamsD = AuxParams<slm_matrix_type::type1, cute::xe4::GMMA::Major::K, TensorDesc, 2>;

  using PostOpPipeline = cutlass::xe4::PipelineTmaAsync<1>;
  using PostOpPipelineState = typename PostOpPipeline::PipelineState;

  using StorePipeline = cutlass::xe4::PipelineTmaAsync<1>;
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

      using FusionStorage = typename FusionCallbacks::SharedStorage;
      FusionStorage thread;
    };
  };

  using TensorStorage = typename SharedStorage::TensorStorage;

  // Host side epilogue arguments
  struct Arguments {
    typename FusionCallbacks::Arguments thread{};
    ElementD const* ptr_D = nullptr;
    StrideD dD{};
  };

  // Device side epilogue params
  struct Params
  {
    using TiledStoreD = decltype(make_xe4_copy<TiledCopyD, AuxParamsD>(
      make_tensor(static_cast<ElementD const*>(nullptr), repeat_like(StrideD{}, int32_t(0)), StrideD{}),
      SmemLayoutD{}, take<0,2>(TileShape{})));

    typename FusionCallbacks::Params thread{};
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
    auto store_d = make_xe4_copy<TiledCopyD, AuxParamsD>(D, SmemLayoutD {}, take<0, 2>(TileShape {}));

    return {
      FusionCallbacks::to_underlying_arguments(problem_shape, args.thread, workspace),
      store_d
    };
  }

  // Note: SharedStorage is unused for CollectiveEpilogue
  CUTLASS_HOST_DEVICE
  CollectiveEpilogue(Params const& params)
      : _params(params) { }

  CUTLASS_DEVICE void
  operator()(
      StorePipeline store_pipeline,
      StorePipelineState& store_pipe_write,
      PostOpPipeline postop_pipeline,
      PostOpPipelineState& postop_pipe_read,
      TensorStorage& shared_tensors,
      uint32_t worker_id)
  {
    postop_pipeline.consumer_try_wait(postop_pipe_read++);

    constexpr auto tile_mn = take<0,2>(TileShape{});
    auto tensor_d = make_tensor(shared_tensors.smem_D.data(), CoreMatrix::retile<ElementD>(tile_mn));

    auto empty_tuple = cute::tuple<>{};
    auto dummy_tensor = make_tensor<float>(Int<1>{});
    auto cst_args = fusion::detail::ConsumerStoreArgs(
      Shape<_512,_512,_512,_1>{},   // ProblemShapeMNKL
      append<3>(TileShape{}, _1{}),      // TileShapeMNK
      make_coord(0,0,0,0),           // TileCoordMNKL
      empty_tuple,      // TiledMma
      Shape<_2,_2>{},   // EpilogueTile
      empty_tuple,      // TiledCopy
      empty_tuple,      // CoordTensor
      empty_tuple,      // Residue
      empty_tuple,      // ThrCoordTensor
      empty_tuple,      // ThrResidue
      dummy_tensor,     // ThrSrcTensor
      worker_id          // thread_idx
    );

    FusionCallbacks fusion_callbacks(_params.thread, shared_tensors.thread);
    auto cst_callbacks = fusion_callbacks.template get_consumer_store_callbacks<true>(cst_args);
    pattern2<FragmentSize, EpiSgNum, SgSize>(cst_callbacks, tensor_d, tensor_d, worker_id);

    store_pipeline.producer_arrive(store_pipe_write++, 1);
  }

  template<
    class ProblemShape,
    class BlockCoordMNL
  >
  CUTLASS_DEVICE void
  store(
      StorePipeline store_pipeline,
      StorePipelineState& store_pipe_state,
      ProblemShape const& problem_shape,
      BlockCoordMNL blk_coord_mnl,
      TensorStorage& shared_tensors)
  {
    store_pipeline.consumer_try_wait(store_pipe_state);       // Wait for all postop threads finish their calculation

    auto sD = make_tensor(shared_tensors.smem_D.data(), SmemLayoutD {});
    auto [M, N, K, L] = problem_shape;
    auto mD_mnl = _params.store_d.get_tma_tensor(make_shape(M, N, L)); // (m,n,l)
    auto gD_mnl = flat_divide(mD_mnl, make_shape(shape<0>(TileShape {}), shape<1>(TileShape {}))); // (BLK_M,BLK_N,m,n,l)
    auto block_store_d = _params.store_d.get_slice(0);
    auto [m_coord, n_coord, l_coord] = blk_coord_mnl;

    auto gD = gD_mnl(_, _, m_coord, n_coord, l_coord);  // (BLK_M,BLK_N)
    auto tDgD = block_store_d.partition_S(gD);           // (TMA,TMA_M,TMA_N)
    auto tDsD = block_store_d.partition_D(sD);    // (TMA,TMA_M,TMA_N)

    auto abar_store = store_pipeline.consumer_get_barrier(store_pipe_state);
    constexpr uint32_t slm_bytes_store = sizeof(ElementD) * size(SmemLayoutD {});
    copy(_params.store_d.with(abar_store), tDsD, tDgD);
    store_pipeline.consumer_commit(store_pipe_state, slm_bytes_store);
    store_pipeline.producer_try_wait(store_pipe_state);
    ++store_pipe_state;
  }

private:
  Params const& _params;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace collective
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////