#pragma once

#include "cutlass/pipeline/xe4_pipeline.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cute/atom/copy_traits_xe4_im2col.hpp"
#include "cute/arch/mma_xe4_amma.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/conv/detail.hpp"
#include "cutlass/arch/arch.h"
#include "cutlass/detail/dependent_false.hpp"
#include "cutlass/conv/collective/detail.hpp"

namespace cutlass::conv::collective {

using namespace cute;
using namespace cute::detail;
using namespace cutlass::gemm;

template <
  class ConvOp_,
  int Stages,
  int NumSpatialDims,
  class ClusterShape,
  class KernelSchedule,
  int PipelineAsyncMmaStages,
  class TileShape_,
  class ElementA_,
  class ElementB_,
  class TiledMma_,
  class SmemLayoutAtomA_,
  class SmemLayoutAtomB_>
struct CollectiveConv
{
  //
  // Type Aliases
  //
  using DispatchPolicy = MainloopXe4DmaGmmaWarpSpecializedImplicitGemm<
    ConvOp_, Stages, NumSpatialDims, ClusterShape, KernelSchedule, PipelineAsyncMmaStages>;
  using TileShape = TileShape_;
  using ElementA = ElementA_;
  using ElementB = ElementB_;
  using TiledMma = TiledMma_;
  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;

  using ElementAccumulator = typename TiledMma::ValTypeC;

  using ArchTag = typename DispatchPolicy::ArchTag;
  static constexpr int NumSpatialDimensions = DispatchPolicy::NumSpatialDimensions;
  static constexpr int NumTensorDimensions = NumSpatialDimensions + 2;

  using StrideA = decltype(cute::Stride<cute::Stride<int64_t, int64_t, int64_t>,cute::Int<1>>{});
  using StrideB = decltype(cute::Stride<int64_t, cute::Stride<cute::Int<1>, int64_t, int64_t>>{});

  static constexpr bool IsRowMajorA = cutlass::detail::is_major<1, StrideA>();
  static constexpr bool IsRowMajorB = cutlass::detail::is_major<0, StrideB>();
  static constexpr slm_matrix_type cm_typeA = IsRowMajorA ? slm_matrix_type::type1 : slm_matrix_type::type2;
  static constexpr slm_matrix_type cm_typeB = slm_matrix_type::type1;
  using TensorDescPtr = uint64_t*;
  using AuxParamsB = AuxParams<cm_typeB, TensorDescPtr, 0>;

  static_assert(Stages >= 2, "Specialization requires Stages set to value 2 or more.");

  using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtomA{}, make_shape(size<0>(TileShape{}), size<2>(TileShape{}), Int<Stages>{}), cute::conditional_t<IsRowMajorA, Step<_2,_1,_3>, Step<_1,_2,_3>>{}));
  using SmemLayoutB = decltype(tile_to_shape(SmemLayoutAtomB{}, make_shape(size<1>(TileShape{}), size<2>(TileShape{}), Int<Stages>{}), cute::conditional_t<!IsRowMajorB, Step<_2,_1,_3>, Step<_1,_2,_3>>{}));

  using MainloopPipeline = cutlass::xe4::PipelineTmaAsync<DispatchPolicy::Stages>;
  using PipelineState  = typename cutlass::xe4::PipelineState<DispatchPolicy::Stages>;

  static constexpr auto ConvOp = ConvOp_::value;
  using ProblemShape = ConvProblemShape<ConvOp, NumSpatialDimensions>;

  static constexpr bool is_im2col_A = true;
  static constexpr bool is_im2col_B = false;

  struct SharedStorage
  {
    struct TensorStorage
    {
      cute::array<ElementA, cute::cosize_v<SmemLayoutA>> smem_A;
      cute::array<ElementB, cute::cosize_v<SmemLayoutB>> smem_B;
    } tensors;
  };
  using TensorStorage = typename SharedStorage::TensorStorage;

  static constexpr uint32_t TmaTransactionBytes =
      (size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) * static_cast<uint32_t>(sizeof(ElementA)))+
      (size<0>(SmemLayoutB{}) * size<1>(SmemLayoutB{}) * static_cast<uint32_t>(sizeof(ElementB)));

  struct Arguments {
    ElementA const* ptr_A {nullptr};
    ElementB const* ptr_B {nullptr};
  };

private:
  template <class TensorA>
  static constexpr auto
  get_tma_load_a_instance(TensorA const& tensor_a, ProblemShape const& problem_shape) {
    // compute the upper and lower corners based on the conv padding
    auto lower_corner_whd = detail::compute_lower_corner_whd(problem_shape);
    auto upper_corner_whd = detail::compute_upper_corner_whd(problem_shape);
    auto lower_srt = detail::compute_lower_srt(problem_shape);

    // The calculation of gbasis strides for dgrad kernel needs perform negate for dilation values.
    cute::array<int32_t, NumSpatialDimensions> stride_srt{};
    for (int i = 0; i < NumSpatialDimensions; ++i) {
      stride_srt[i] = ConvOp == conv::Operator::kDgrad ?
        -problem_shape.dilation[NumSpatialDimensions-1-i] :
        problem_shape.dilation[NumSpatialDimensions-1-i];
    }

    return make_im2col_tma_copy<cute::xe4::ASYNC_ROW_LOAD_IM2COL, cute::C<cm_typeA>>(
      tensor_a,
      make_layout(make_shape(size<0>(TileShape{}), size<2>(TileShape{})),
        make_stride(size<2>(TileShape{}), Int<1>{})),
      Layout<Shape<cute::C<LANESIZE>, _1>>{},
      make_layout(make_shape(Int<1>{}, size<2>(TileShape{}))),
      make_shape(size<0>(TileShape{}), size<2>(TileShape{})),
      1,
      shape(lower_corner_whd),
      shape(upper_corner_whd),
      cute::reverse(shape(problem_shape.lower_padding)),
      cute::reverse(shape(problem_shape.upper_padding)),
      cute::reverse(shape(problem_shape.traversal_stride)),
      shape(lower_srt),
      shape(stride_srt)
    );
  }

  // Get tma_load_b instantce.
  template <class TensorB>
  static constexpr auto
  get_tma_load_b_instance(TensorB const& tensor_b, ProblemShape const& problem_shape) {
    auto layoutSB = make_layout(make_shape(size<1>(TileShape{}), size<2>(TileShape{}), Int<Stages>{}),
                                make_stride(size<2>(TileShape{}), Int<1>{}, size<1>(TileShape{}) * size<2>(TileShape{})));
    return make_xe4_copy_conv2d<cute::xe4::ASYNC_TENSOR_LOAD, AuxParamsB>(
      tensor_b,
      layoutSB(_, _, 0),
      make_shape(size<1>(TileShape{}), make_shape(size<2>(TileShape{})))
    );
  }

public:
  // Performs im2col transformations on the input of type ConvProblemShape
  static constexpr auto
  get_problem_shape_MNKL(ProblemShape const& problem_shape) {
    if constexpr (is_im2col_A || is_im2col_B) {
      // transformation + im2col linearization
      return cutlass::conv::detail::get_linearized_problem_shape_MNKL(problem_shape);
    }
    else {
      // transformation
      return cutlass::conv::detail::get_transformed_problem_shape_MNKL(problem_shape);
    }
  }

  // Device side kernel params
  struct Params {
    using _Submode = decltype(take<0, NumTensorDimensions - 1>(typename ProblemShape::TensorExtent{}));
    using TensorShapeA = decltype(make_shape(_Submode{}, int(0)));
    using TensorShapeB = decltype(repeat_like(StrideB{}, int32_t(0)));

    using TMA_A = decltype(get_tma_load_a_instance(
      make_tensor(
        static_cast<ElementA const*>(nullptr),
        make_layout(TensorShapeA{}, StrideA{})),
      ConvProblemShape<ConvOp, NumSpatialDimensions>{}));

    using TMA_B = decltype(get_tma_load_b_instance(
      make_tensor(
        static_cast<ElementB const*>(nullptr),
        make_layout(TensorShapeB{}, StrideB{})),
      ConvProblemShape<ConvOp, NumSpatialDimensions>{}));

    // Members
    TMA_A tma_load_a;
    TMA_B tma_load_b;
    uint32_t tma_transaction_bytes = TmaTransactionBytes;
  };

  //
  //  Methods
  //
  // Lowers the host side user facing arguments to the kernel facing lauch params
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args) {
    // from the flat problem shape arrays of ConvProblemShape<ConvOp, N>, create a rank-3 MNK problem shape tuple
    // tma desc creation depends on the original untransformed domain.

    // A extents.
    auto shape_A_orig = problem_shape.get_shape_A();
    // B extents.
    auto shape_B_orig = problem_shape.get_shape_B();

    // Fill inferred cute strides from flat stride arrays
    auto dA = make_cute_packed_stride(StrideA{}, problem_shape.stride_A, ConvOp);
    auto dB = make_cute_packed_stride(StrideB{}, problem_shape.stride_B, ConvOp);

    Tensor tensor_a = make_tensor(args.ptr_A, make_layout(shape_A_orig, dA));
    Tensor tensor_b = make_tensor(args.ptr_B, make_layout(shape_B_orig, dB));

    auto tma_load_a = get_tma_load_a_instance(tensor_a, problem_shape);
    auto tma_load_b = get_tma_load_b_instance(tensor_b, problem_shape);

    return {
      tma_load_a,
      tma_load_b,
      TmaTransactionBytes
    };
  }

  template <class ProblemShapeMNKL>
  CUTLASS_DEVICE auto
  load_init(ProblemShapeMNKL const& problem_shape_MNKL, Params const& mainloop_params){
    using X = Underscore;
    // Separate out problem shape for convenience
    auto [M, N, K, L] = problem_shape_MNKL;

    // TMA requires special handling of strides to deal with coord codomain mapping
    // Represent the full tensors -- get these from TMA
    Tensor mA_mk = mainloop_params.tma_load_a.get_tma_tensor(make_shape(M,K));                            // (m,k)
    Tensor mB_nk = mainloop_params.tma_load_b.get_tma_tensor(make_shape(N,K));                            // (n,k)

    // Make tiled views, defer the slice
    Tensor gA_mk = local_tile(mA_mk, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});        // (BLK_M,BLK_K,m,k)
    Tensor gB_nk = local_tile(mB_nk, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});        // (BLK_N,BLK_K,n,k)

    return cute::make_tuple(gA_mk, gB_nk);
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  /// Producer Perspective
  template <
    class TensorA, class TensorB,
    class KTileIterator, class BlockCoord
  >
  CUTLASS_DEVICE void
  load(
    Params const& mainloop_params,
    MainloopPipeline pipeline,
    PipelineState smem_pipe_producer_state,
    cute::tuple<TensorA, TensorB> const& load_inputs,
    BlockCoord const& blk_coord,
    KTileIterator k_tile_iter, int k_tile_count,
    int thread_idx,
    TensorStorage& shared_tensors) {

    auto sA = make_tensor(shared_tensors.smem_A.data(), SmemLayoutA {});
    auto sB = make_tensor(shared_tensors.smem_B.data(), SmemLayoutB {});

    auto thr_load_a = mainloop_params.tma_load_a.get_slice(thread_idx);
    auto block_load_b = mainloop_params.tma_load_b.get_slice(get_wgid<0>());

    auto [gA_mk, gB_nk] = load_inputs;

    // Partition the inputs based on the current block coordinates.
    auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;

    Tensor gA = gA_mk(_,_,m_coord,_);                                                     // (BLK_M,BLK_K,k)
    Tensor gB = gB_nk(_,_,n_coord,_);                                                     // (BLK_N,BLK_K,k)

    // Applies the mapping from block_tma_a
    Tensor tAgA = thr_load_a.partition_S(gA);                                                 // (TMA,TMA_M,TMA_K,k)
    Tensor tAsA = thr_load_a.partition_D(sA);                                              // (TMA,TMA_M,TMA_K,PIPE)

    Tensor tBgB = block_load_b.partition_S(gB);                                                 // (TMA,TMA_N,TMA_K,k)
    Tensor tBsB = block_load_b.partition_D(sB);                                              // (TMA,TMA_N,TMA_K,PIPE)

    // Mainloop
    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 0; --k_tile_count) {
      uint32_t write_stage = smem_pipe_producer_state.index();
      auto abar_prod = pipeline.producer_get_barrier(write_stage);

      pipeline.producer_try_wait(smem_pipe_producer_state);
      copy(mainloop_params.tma_load_a.with(abar_prod), tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,write_stage));

      if (thread_idx == 0) {
        pipeline.producer_commit(write_stage, mainloop_params.tma_transaction_bytes);
        copy(mainloop_params.tma_load_b.with(abar_prod), tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,write_stage));
      }

      ++k_tile_iter;
      ++smem_pipe_producer_state;
    }
  }

  template <class Pipeline, class PipelineState, class FinalPipeline, class FinalPipelineState, class FrgTensorAcc, class FrgTensorC,
    class SlmPtr>
  CUTLASS_DEVICE void
  mma(Pipeline pipeline, PipelineState slm_pipe_read, FinalPipeline finalPipeline, FinalPipelineState& finalPipelineState,
    FrgTensorAcc& accumulator, FrgTensorC& sC, int k_tile_count, int local_id, SlmPtr slm_ptr) {
    auto shared_tensors = reinterpret_cast<TensorStorage*>(slm_ptr);
    auto sA = make_tensor(reinterpret_cast<ElementA *>(shared_tensors->smem_A.data()), SmemLayoutA {});
    auto sB = make_tensor(reinterpret_cast<ElementB *>(shared_tensors->smem_B.data()), SmemLayoutB {});

    TiledMma tiled_mma;
    auto thread_mma = tiled_mma.get_thread_slice(0);
    auto tCrA = thread_mma.partition_fragment_A(sA);            // (MMA,MMA_M,MMA_K,PIPE)
    auto tCrB = thread_mma.partition_fragment_B(sB);            // (MMA,MMA_N,MMA_K,PIPE)
    auto accum = thread_mma.partition_fragment_C(accumulator);  // (MMA,MMA_M,MMA_N)
    auto tCrC = thread_mma.partition_fragment_C(sC);            // (MMA,MMA_M,MMA_N)

    constexpr auto scaleOutOne = cute::C<cute::xe4::GMMA::ScaleOut::One>{};
    constexpr auto scaleOutZero = cute::C<cute::xe4::GMMA::ScaleOut::Zero>{};
    constexpr auto dstIsAccum = cute::C<cute::xe4::GMMA::DstType::Accum>{};
    constexpr auto dstIsMatC = cute::C<cute::xe4::GMMA::DstType::MatC>{};

    pipeline.consumer_try_wait(slm_pipe_read);
    auto abar_cons_base = pipeline.abar_cons_base;

    if (k_tile_count == 1) {
      cute::gemm(tiled_mma.with(scaleOutZero, dstIsMatC, abar_cons_base), tCrC, tCrA(_,_,_,0), tCrB(_,_,_,0), accum);
      pipeline.consumer_commit(slm_pipe_read);
    } else {
      cute::gemm(tiled_mma.with(scaleOutZero, dstIsAccum, abar_cons_base), tCrA(_,_,_,0), tCrB(_,_,_,0), accum);
      pipeline.consumer_commit(slm_pipe_read);

      for (uint32_t i = 1; i < k_tile_count - 1; i++) {
        ++slm_pipe_read;
        uint32_t abar_index = slm_pipe_read.index();
        auto abar_cons = pipeline.consumer_get_barrier(abar_index);
        pipeline.consumer_try_wait(slm_pipe_read);
        cute::gemm(tiled_mma.with(scaleOutOne, dstIsAccum, abar_cons), tCrA(_,_,_,abar_index), tCrB(_,_,_,abar_index), accum);
        pipeline.consumer_commit(slm_pipe_read);
      }
      {
        uint32_t abar_store_prod_index = finalPipelineState.index();
        auto abar_store_prod = finalPipeline.producer_get_barrier(abar_store_prod_index);

        ++slm_pipe_read;
        uint32_t abar_index = slm_pipe_read.index();

        uint32_t phase = ((k_tile_count - 1) / Stages) & 1u;
        pipeline.consumer_try_wait(abar_index, phase);
        cute::gemm(tiled_mma.with(scaleOutOne, dstIsMatC, abar_store_prod), tCrC, tCrA(_,_,_,abar_index), tCrB(_,_,_,abar_index), accum);
        finalPipeline.producer_commit(finalPipelineState, 1);
      }
    }
  }
};
} // namespace cutlass::conv::collective
