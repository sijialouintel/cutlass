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

constexpr slm_matrix_type cm_typeD = slm_matrix_type::type1;
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Applies an element wise operation to all elements within the fragment
/// and writes them out to destination storage.
template <
  conv::Operator ConvOp,
  int NumSpatialDims,
  class SmemLayoutD_,
  class TileShape_,
  class ElementD_
>
class EpilogueConv {
public:
  using ElementD = ElementD_;
  using TileShape = TileShape_;
  using SmemLayoutD = SmemLayoutD_;

  using EpilogueStorePipeline = cutlass::xe4::PipelineTmaAsync<1, 1>;
  using StorePipelineState = typename cutlass::xe4::PipelineState<1>;

  using StrideC = decltype(cute::Stride<cute::Stride<int64_t, int64_t, int64_t>,cute::Int<1>>{});
  static constexpr int NumTensorDimensions = NumSpatialDims + 2;

  struct SharedStorage
  {
    struct TensorStorage
    {
      cute::array<ElementD, cute::cosize_v<SmemLayoutD>> smem_D;
    };
  };

  using TensorStorage = typename SharedStorage::TensorStorage;

  static constexpr uint32_t TmaTransactionBytes =
    (size<0>(SmemLayoutD{}) * size<1>(SmemLayoutD{}) * static_cast<uint32_t>(sizeof(ElementD)));

  // Host side epilogue arguments
  struct Arguments {
    ElementD const* ptr_D = nullptr;
  };

  // Device side epilogue params
  template <class TensorD>
  static constexpr auto
  get_tma_store_d_instance(TensorD const& tensor_d)
  {
    return make_im2col_tma_copy<cute::xe4::ASYNC_ROW_STORE_IM2COL, cute::C<cm_typeD>>(tensor_d,
      make_layout(make_shape(shape<0>(TileShape{}), shape<1>(TileShape{})),
                  make_stride(shape<1>(TileShape{}), Int<1>{})),
      Layout<Shape<cute::C<LANESIZE>, _1>>{},
      make_layout(make_shape(Int<1>{}, shape<1>(TileShape{}))),
      make_shape(shape<0>(TileShape{}), shape<1>(TileShape{})),
      1,
      append<2>(Stride<_0>{}, Int<0>{}),
      append<2>(Stride<_0>{}, Int<0>{}),
      append<2>(Stride<_0>{}, Int<0>{}),
      append<2>(Stride<_0>{}, Int<0>{}),
      append<2>(Stride<_1>{}, Int<1>{}),
      append<2>(Stride<_0>{}, Int<0>{}),
      append<2>(Stride<_1>{}, Int<1>{})
    );
  }

  struct Params
  {
    static constexpr int RankT = NumSpatialDims + 2;
    using TensorExtent  = cute::array<int, RankT>;
    using _Submode = decltype(take<0, NumSpatialDims + 1>(TensorExtent{}));
    using TensorShapeD = decltype(make_shape(_Submode{}, int(0)));

    using TMA_D = decltype(get_tma_store_d_instance(
      make_tensor(
        static_cast<ElementD const*>(nullptr),
        make_layout(TensorShapeD{}, StrideC{}))));

    TMA_D tma_store_d;
    uint32_t tma_transaction_bytes = TmaTransactionBytes;
  };

  //
  // Methods
  //
  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args) {
    auto shape_D_orig = problem_shape.get_shape_C();
    auto dD = make_cute_packed_stride(StrideC{}, problem_shape.stride_C, ConvOp);

    Tensor tensor_d = make_tensor(args.ptr_D, make_layout(shape_D_orig, dD));

    auto tma_store_d = get_tma_store_d_instance(tensor_d);

    return {tma_store_d, TmaTransactionBytes};
  }

  template<class ProblemShapeMNKL>
  CUTLASS_DEVICE void
  store(
    Params const& epilogue_params,
    EpilogueStorePipeline epilogue_store_pipeline,
    StorePipelineState pipe_store_state,
    ProblemShapeMNKL const& problem_shape_MNKL,
    int thread_idx,
    TensorStorage& shared_tensors)
  {
    auto sD = make_tensor(reinterpret_cast<ElementD *>(shared_tensors.smem_D.data()), SmemLayoutD {});

    auto [M, N, K, L] = problem_shape_MNKL;
    Tensor mD_mn = epilogue_params.tma_store_d.get_tma_tensor(make_shape(M,N));
    uint32_t wgid_x = get_wgid<0>();
    uint32_t wgid_y = get_wgid<1>();

    auto thr_store_d = epilogue_params.tma_store_d.get_slice(thread_idx);
    Tensor gD_mn = local_tile(mD_mn, TileShape{}, make_coord(_,_,_), Step<_1, _1, X>{});
    Tensor gD = gD_mn(_,_,wgid_y,wgid_x);

    Tensor tDsD = thr_store_d.partition_S(sD);
    Tensor tDgD = thr_store_d.partition_D(gD);

    epilogue_store_pipeline.consumer_try_wait(pipe_store_state);
    if(thread_idx == 0) {
        epilogue_store_pipeline.consumer_commit(pipe_store_state, epilogue_params.tma_transaction_bytes);
    }

    uint32_t abar_store_cons_index = pipe_store_state.index();
    auto abar_store = epilogue_store_pipeline.producer_get_barrier(abar_store_cons_index);
    copy(epilogue_params.tma_store_d.with(abar_store), tDsD, tDgD);
    ++pipe_store_state;
    epilogue_store_pipeline.producer_try_wait(pipe_store_state);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace collective
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////