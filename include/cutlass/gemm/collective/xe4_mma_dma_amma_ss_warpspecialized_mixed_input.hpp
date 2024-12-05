#pragma once

#include "cute/atom/mma_traits_xe4_amma.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/util/packed_stride.hpp"

namespace cutlass::gemm::collective {

using namespace cute;
using namespace cute::detail;
using namespace cutlass::gemm;

template <
  int Stages,
  uint32_t SplitB,
  uint32_t MxScaleSize,
  class ClusterShape,
  class KernelSchedule,
  class TileShape_,
  class ElementATuple,
  class StrideA_,
  class ElementBTuple,
  class StrideB_,
  class TiledMma_,
  class GmemTiledCopyA_,
  class SmemLayoutAtomA_,
  class SmemCopyAtomA_,
  class TransformA_,
  class GmemTiledCopyB_,
  class SmemLayoutAtomB_,
  class SmemCopyAtomB_,
  class TransformB_>
struct CollectiveMma<
  MainloopXe4DmaGmmaWarpSpecializedMixedInput<Stages, SplitB, MxScaleSize, ClusterShape, KernelSchedule>,
  TileShape_,
  ElementATuple,
  StrideA_,
  ElementBTuple,
  StrideB_,
  TiledMma_,
  GmemTiledCopyA_,
  SmemLayoutAtomA_,
  SmemCopyAtomA_,
  TransformA_,
  GmemTiledCopyB_,
  SmemLayoutAtomB_,
  SmemCopyAtomB_,
  TransformB_>
{
  using DispatchPolicy = MainloopXe4DmaGmmaWarpSpecializedMixedInput<Stages, SplitB, MxScaleSize, ClusterShape, KernelSchedule>;
  using TileShape = TileShape_;
  using TiledMma = TiledMma_;
  using ElementA = cute::tuple_element_t<0, ElementATuple>;
  using ElementMetaA = cute::tuple_element_t<1, ElementATuple>;
  using ElementAMma = typename TiledMma::ValTypeA;
  using StrideA = StrideA_;
  using ElementB = cute::tuple_element_t<0, ElementBTuple>;
  using ElementMetaB = cute::tuple_element_t<1, ElementATuple>;
  using ElementBMma = typename TiledMma::ValTypeB;
  using StrideB = StrideB_;
  using GmemTiledCopyA = GmemTiledCopyA_;
  using GmemTiledCopyB = GmemTiledCopyB_;
  using GmemTiledCopyScale = cute::xe4::ASYNC_TENSOR_LOAD;
  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;
  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB = SmemCopyAtomB_;

  using TensorDesc = uint64_t*;
  using MatrixDesc = uint32_t;
  using Abarrier = typename TiledMma::AbarrierType;
  using ElementScale = typename TiledMma::ValTypeE;
  using ElementAccumulator = typename TiledMma::ValTypeC;
  using StrideScale = cute::Stride<cute::Int<1>, int64_t, int64_t>;

  static constexpr cute::xe4::GMMA::Major tnspA = TiledMma::tnspA;
  static constexpr cute::xe4::GMMA::Major tnspB = TiledMma::tnspB;
  static constexpr cute::xe4::GMMA::Major tnspScale = cute::xe4::GMMA::Major::MN;

  using AuxParamsA = AuxParams<(tnspA == cute::xe4::GMMA::Major::K ? slm_matrix_type::type1 : slm_matrix_type::type2), tnspA, TensorDesc, 0>;
  using AuxParamsB = AuxParams<slm_matrix_type::type1, tnspB, TensorDesc, 1>;
  using AuxParamsMetaA = AuxParams<slm_matrix_type::type2, tnspScale, TensorDesc, 3, true>;
  using AuxParamsMetaB = AuxParams<slm_matrix_type::type2, tnspScale, TensorDesc, 4, true>;

  using MainloopPipeline = cutlass::xe4::PipelineTmaAsync<Stages, 0, Abarrier>;
  using PipelineState = typename MainloopPipeline::PipelineState;

  static constexpr uint32_t StagesB = DispatchPolicy::StagesB;
  using MainloopPipelineB = cutlass::xe4::PipelineTmaAsync<StagesB, 1, Abarrier>;
  using PipelineStateB = typename MainloopPipelineB::PipelineState;

  using SplitBTileShape = decltype(shape_div(TileShape{}, Shape<_1, Int<SplitB>, _1>{}));

  static_assert(DispatchPolicy::Stages >= 2, "Specialization requires Stages set to value 2 or more.");
  static_assert(cute::is_same_v<GmemTiledCopyA, cute::xe4::ASYNC_TENSOR_LOAD> || cute::is_same_v<GmemTiledCopyA, cute::xe4::ASYNC_TENSOR_LOAD_MULTICAST>,
      "GmemTiledCopy - invalid XE4 DMA copy atom specified.");
  static_assert(cute::is_same_v<GmemTiledCopyB, cute::xe4::ASYNC_TENSOR_LOAD> || cute::is_same_v<GmemTiledCopyB, cute::xe4::ASYNC_TENSOR_LOAD_MULTICAST>,
      "GmemTiledCopy - invalid XE4 DMA copy atom specified.");

  static_assert(cute::rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M,K)");
  static_assert((size<0>(SplitBTileShape{}) % size<0>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(SplitBTileShape{}) % size<1>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(cute::rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (N,K)");
  static_assert((size<1>(SplitBTileShape{}) % size<0>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(SplitBTileShape{}) % size<1>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  // Tile along modes in a way that maximizes the TMA box size.
  using SmemLayoutA = decltype(tile_to_shape(
    SmemLayoutAtomA{},
    append(select<0,2>(SplitBTileShape{}), Int<Stages>{}),
    cute::conditional_t<tnspA == cute::xe4::GMMA::Major::K, Step<_2,_1,_3>, Step<_1,_2,_3>>{}));

  using SmemLayoutB = decltype(tile_to_shape(
    SmemLayoutAtomB{},
    append(select<1,2>(SplitBTileShape{}), Int<StagesB>{}),
    cute::conditional_t<tnspB == cute::xe4::GMMA::Major::K, Step<_2,_1,_3>, Step<_1,_2,_3>>{}));

  // for uint8_t scale tensor, we divide K dim by 32
  using SmemLayoutAtomScale = Layout<Shape<_32, _1>, Stride<_1, _32>>;
  using ScaleTileShape = decltype(shape_div(TileShape{}, Shape<_1, Int<SplitB>, _32>{}));

  // for smem scale tensor, the K dim is rounded up to 32, and have different tile shape for partitioning
  using ScaleTileK = decltype(get<2>(ScaleTileShape{}));
  using ScaleTileKRoundUp = decltype(round_up(ScaleTileK{}, Int<MxScaleSize>{}));
  using SmemScaleTileShape = decltype(append(take<0,2>(ScaleTileShape{}), ScaleTileKRoundUp{}));
  using TVLayoutMetaA = cute::xe4::ABLayout<get<0>(SmemScaleTileShape{}), get<2>(SmemScaleTileShape{})>;
  using TVLayoutMetaB = cute::xe4::ABLayout<get<1>(SmemScaleTileShape{}), get<2>(SmemScaleTileShape{})>;

  // MetaA is a column-major WgM x MetaK matrix
  using SmemLayoutMetaA = decltype(tile_to_shape(
    SmemLayoutAtomScale{},
    append(select<0,2>(SmemScaleTileShape{}), Int<Stages>{})));

  // MetaB is a column-major WgN x MetaK matrix
  using SmemLayoutMetaB = decltype(tile_to_shape(
    SmemLayoutAtomScale{},
    append(select<1,2>(SmemScaleTileShape{}), Int<StagesB>{})));

  struct SharedStorage
  {
    struct TensorStorage
    {
      cute::ArrayEngine<ElementAMma, cute::cosize_v<SmemLayoutA>> smem_A;
      cute::ArrayEngine<ElementBMma, cute::cosize_v<SmemLayoutB>> smem_B;
      cute::array<ElementScale, cute::cosize_v<SmemLayoutMetaA>> smem_metaA;
      cute::array<ElementScale, cute::cosize_v<SmemLayoutMetaB>> smem_metaB;
    };
  };

  using TensorStorage = typename SharedStorage::TensorStorage;

  static constexpr uint32_t TransactionBytes_A =
        cutlass::bits_to_bytes(cosize(take<0,2>(SmemLayoutA{})) * cute::sizeof_bits_v<ElementAMma>) +
        uint32_t(cosize(take<0,2>(SmemLayoutMetaA{}))) * sizeof(ElementScale);

  static constexpr uint32_t TransactionBytes_B =
        cutlass::bits_to_bytes(cosize(take<0,2>(SmemLayoutB{})) * cute::sizeof_bits_v<ElementBMma>) +
        uint32_t(cosize(take<0,2>(SmemLayoutMetaB{}))) * sizeof(ElementScale);

  // Host side kernel arguments
  struct Arguments {
    ElementA const* ptr_A {nullptr};
    StrideA dA;
    ElementB const* ptr_B {nullptr};
    StrideB dB;
    ElementMetaA const* ptr_metaA {nullptr};
    StrideScale dMetaA;
    ElementMetaB const* ptr_metaB {nullptr};
    StrideScale dMetaB;
    int metaK = 0;
  };

  // Device side kernel params
  struct Params {
    using TiledLoadA = decltype(make_xe4_copy<GmemTiledCopyA, AuxParamsA, ElementA>(
      make_tensor(recast_ptr<ElementAMma>(nullptr), repeat_like(StrideA{}, int32_t(0)), StrideA{}),
      SmemLayoutA{}(_, _, _0{}), select<0,2>(SplitBTileShape{}), size<1>(ClusterShape{})));

    using TiledLoadB = decltype(make_xe4_copy<GmemTiledCopyB, AuxParamsB, ElementB>(
      make_tensor(recast_ptr<ElementBMma>(nullptr), repeat_like(StrideB{}, int32_t(0)), StrideB{}),
      SmemLayoutB{}(_, _, _0{}), select<1,2>(SplitBTileShape{}), size<0>(ClusterShape{})));

    using TiledLoadMetaA = decltype(make_xe4_copy<GmemTiledCopyA, AuxParamsMetaA, ElementMetaA>(
      make_tensor(recast_ptr<ElementScale>(nullptr), repeat_like(StrideScale{}, int32_t(0)), StrideScale{}),
      make_layout(select<0,2>(ScaleTileShape{})), select<0,2>(ScaleTileShape{}), size<1>(ClusterShape{})));

    using TiledLoadMetaB = decltype(make_xe4_copy<GmemTiledCopyB, AuxParamsMetaB, ElementMetaB>(
      make_tensor(recast_ptr<ElementScale>(nullptr), repeat_like(StrideScale{}, int32_t(0)), StrideScale{}),
      make_layout(select<1,2>(ScaleTileShape{})), select<1,2>(ScaleTileShape{}), size<0>(ClusterShape{})));

    TiledLoadA load_a;
    TiledLoadB load_b;
    TiledLoadMetaA load_meta_a;
    TiledLoadMetaB load_meta_b;
    int metaK = 0;
  };

  template<class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args) {
    // Optionally append 1s until problem shape is rank-4 (MNKL), in case it is only rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M, N, K, L] = problem_shape;

    auto ptr_A = recast_ptr<ElementAMma>(args.ptr_A);
    auto ptr_B = recast_ptr<ElementBMma>(args.ptr_B);
    auto ptr_metaA = recast_ptr<ElementScale>(args.ptr_metaA);
    auto ptr_metaB = recast_ptr<ElementScale>(args.ptr_metaB);

    Tensor tensor_a = make_tensor(ptr_A, make_layout(make_shape(M,K,L), args.dA));
    Tensor tensor_b = make_tensor(ptr_B, make_layout(make_shape(N,K,L), args.dB));

    Tensor tensor_meta_a = make_tensor(ptr_metaA, make_layout(make_shape(M,args.metaK,L), args.dMetaA));
    Tensor tensor_meta_b = make_tensor(ptr_metaB, make_layout(make_shape(N,args.metaK,L), args.dMetaB));

    auto load_a = make_xe4_copy<GmemTiledCopyA, AuxParamsA, ElementA>(
      tensor_a,
      SmemLayoutA{}(_, _, _0{}),
      select<0,2>(SplitBTileShape{}),
      size<1>(ClusterShape{}));   // mcast along N mode for this M load, if any

    auto load_b = make_xe4_copy<GmemTiledCopyB, AuxParamsB, ElementB>(
      tensor_b,
      SmemLayoutB{}(_, _, _0{}),
      select<1,2>(SplitBTileShape{}),
      size<0>(ClusterShape{}));   // mcast along M mode for this N load, if any

    auto load_meta_a = make_xe4_copy<GmemTiledCopyA, AuxParamsMetaA, ElementMetaA>(
      tensor_meta_a,
      make_layout(select<0,2>(ScaleTileShape{})), select<0,2>(ScaleTileShape{}),
      size<1>(ClusterShape{})); // mcast along N mode for this M load, if any

    auto load_meta_b = make_xe4_copy<GmemTiledCopyB, AuxParamsMetaB, ElementMetaB>(
      tensor_meta_b,
      make_layout(select<1,2>(ScaleTileShape{})), select<1,2>(ScaleTileShape{}),
      size<0>(ClusterShape{}));   // mcast along M mode for this N load, if any

    return {load_a, load_b, load_meta_a, load_meta_b, args.metaK};
  }

  template <class ProblemShape>
  CUTLASS_DEVICE auto
  load_init(ProblemShape const& problem_shape, Params const& mainloop_params) const {
    auto [M, N, K, L] = problem_shape;

    auto mA_mkl = mainloop_params.load_a.get_tma_tensor(make_shape(M, K, L));   // (m,k,l)
    auto mB_nkl = mainloop_params.load_b.get_tma_tensor(make_shape(N, K, L));   // (n,k,l)

    const auto metaK = mainloop_params.metaK;
    auto mMetaA_mkl = mainloop_params.load_meta_a.get_tma_tensor(make_shape(M, metaK, L));   // (m,meta_k,l)
    auto mMetaB_nkl = mainloop_params.load_meta_b.get_tma_tensor(make_shape(N, metaK, L));   // (n,meta_k,l)

    auto gA_mkl = local_tile(mA_mkl, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});        // (BLK_M,BLK_K,m,k,l)
    auto gB_nkl = local_tile(mB_nkl, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});        // (BLK_N,BLK_K,n,k,l)

    auto gMetaA_mkl = local_tile(mMetaA_mkl, ScaleTileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});   // (BLK_M,META_K,m,k,l)
    auto gMetaB_nkl = flat_divide(mMetaB_nkl, make_tile(get<1>(TileShape{}), get<2>(ScaleTileShape{})));   // (BLK_N,META_K,n,k,l)

    return cute::make_tuple(gA_mkl, gB_nkl, gMetaA_mkl, gMetaB_nkl);
  }

  CUTLASS_DEVICE static auto
  calculateClusterMasks() {
    uint32_t cluster_wgid_x = get_cluster_wgid<0>();
    uint32_t cluster_wgid_y = get_cluster_wgid<1>();

    uint32_t coop_set_id_a = cluster_wgid_y;
    constexpr uint32_t cluster_size_x = size<1>(ClusterShape{});
    uint32_t cluster_mask_a = ((1u << cluster_size_x) - 1) << (coop_set_id_a * cluster_size_x);

    uint32_t coop_set_id_b = cluster_wgid_x;
    uint32_t cluster_mask_b_base = 1u << coop_set_id_b;
    constexpr uint32_t cluster_size_y = size<0>(ClusterShape{});
    constexpr uint32_t cluster_mask_b_scale = ((1u << (cluster_size_x * cluster_size_y)) - 1) / ((1u << cluster_size_x) - 1);
    uint32_t cluster_mask_b = cluster_mask_b_base * cluster_mask_b_scale;

    return cute::make_tuple(cluster_mask_a, cluster_mask_b);
  }

  template <class TensorA, class TensorB, class MetaA, class MetaB, class BlockCoord, class ClusterMask>
  CUTLASS_DEVICE void
  load(Params const& mainloop_params, MainloopPipeline pipeline, PipelineState slm_pipe_write, MainloopPipelineB pipeline_b, PipelineStateB slm_pipe_write_b,
    cute::tuple<TensorA, TensorB, MetaA, MetaB> const& load_inputs, BlockCoord const& blk_coord, int k_tile_count, int local_id, ClusterMask const& cluster_mask, TensorStorage& shared_tensors) {

    TiledMma tiled_mma;
    Copy_Atom<AutoVectorizingCopy, uint32_t> copy_atom_scale;
    auto load_smem_meta_a = make_tiled_copy_E(copy_atom_scale, tiled_mma, TVLayoutMetaA{}, select<0, 2>(SmemScaleTileShape{}));
    auto load_smem_meta_b = make_tiled_copy_E(copy_atom_scale, tiled_mma, TVLayoutMetaB{}, select<1, 2>(SmemScaleTileShape{}));

    auto sA = make_tensor(shared_tensors.smem_A.begin(), SmemLayoutA {});
    auto sB = make_tensor(shared_tensors.smem_B.begin(), SmemLayoutB {});
    auto sMetaA = make_tensor(shared_tensors.smem_metaA.data(), SmemLayoutMetaA {});
    auto sMetaB = make_tensor(shared_tensors.smem_metaB.data(), SmemLayoutMetaB {});

    auto [cluster_mask_a, cluster_mask_b] = cluster_mask;

    uint32_t cluster_wgid_x = get_cluster_wgid<0>();
    uint32_t cluster_wgid_y = get_cluster_wgid<1>();

    auto block_load_a = mainloop_params.load_a.get_slice(0);
    auto block_load_b = mainloop_params.load_b.get_slice(0);
    auto block_load_meta_a = mainloop_params.load_meta_a.get_slice(0);
    auto block_load_meta_b = mainloop_params.load_meta_b.get_slice(0);
    auto block_load_smem_meta_a = load_smem_meta_a.get_thread_slice(0);
    auto block_load_smem_meta_b = load_smem_meta_b.get_thread_slice(0);

    auto [gA_mkl, gB_nkl, gMetaA_mkl, gMetaB_nkl] = load_inputs;
    auto [m_coord, n_coord, l_coord] = blk_coord;

    auto gA = gA_mkl(_, _, m_coord, _, l_coord);        // (BLK_M,BLK_K,k)
    auto tAgA = block_load_a.partition_S(gA);           // (TMA,TMA_M,TMA_K,k)
    auto tAsA = block_load_a.partition_D(sA);           // (TMA,TMA_M,TMA_K,PIPE)

    auto gB = gB_nkl(_, _, n_coord, _, l_coord);        // (BLK_N,BLK_K,k)
    auto tBgB = block_load_b.partition_S(gB);           // (TMA,TMA_N,TMA_K,k)
    auto tBsB = block_load_b.partition_D(sB);           // (TMA,TMA_N,TMA_K,PIPE)

    auto gMetaA = gMetaA_mkl(_, _, m_coord, _, l_coord);      // (BLK_M,BLK_K,k)
    auto tAgMetaA = block_load_meta_a.partition_S(gMetaA);    // (TMA,TMA_M,TMA_K,k)
    auto tAsMetaA = block_load_smem_meta_a.partition_D(sMetaA);    // (TMA,TMA_M,TMA_K,PIPE)

    auto gMetaB = gMetaB_nkl(_, _, n_coord, _, l_coord);      // (BLK_N,BLK_K,k)
    auto tBgMetaB = block_load_meta_b.partition_S(gMetaB);    // (TMA,TMA_N,TMA_K,k)
    auto tBsMetaB = block_load_smem_meta_b.partition_D(sMetaB);    // (TMA,TMA_N,TMA_K,PIPE)

    for (int i = 0; i < k_tile_count; ++i, ++slm_pipe_write) {
      pipeline.producer_try_wait(slm_pipe_write);

      uint32_t write_stage = slm_pipe_write.index();
      auto abar_prod = pipeline.producer_get_barrier(slm_pipe_write);

      constexpr auto dimIndex = _2{};
      const uint32_t newDimSize = (i+1) * uint32_t(ScaleTileK{});  // set dim size to make dma load meta with OOB
      copy(mainloop_params.load_a.with(abar_prod, cluster_mask_a), tAgA(_,_,_,i), tAsA(_,_,_,write_stage));
      copy(mainloop_params.load_meta_a.with(dimIndex, newDimSize, abar_prod), tAgMetaA(_,_,_,i), tAsMetaA(_,_,_,write_stage));
      pipeline.producer_commit(slm_pipe_write, TransactionBytes_A);

      for (int bIdx = 0; bIdx < SplitB; ++bIdx, ++slm_pipe_write_b) {
        pipeline_b.producer_try_wait(slm_pipe_write_b);
        uint32_t write_stage_b = slm_pipe_write_b.index();
        auto abar_prod_b = pipeline_b.producer_get_barrier(slm_pipe_write_b);
        copy(mainloop_params.load_b.with(abar_prod_b, cluster_mask_b), tBgB(_,bIdx,_,i), tBsB(_,_0{},_,write_stage_b));
        copy(mainloop_params.load_meta_b.with(dimIndex, newDimSize, abar_prod_b), tBgMetaB(_,bIdx,_,i), tBsMetaB(_,_0{},_,write_stage_b));
        pipeline_b.producer_commit(slm_pipe_write_b, TransactionBytes_B);
      }
    }
  }

  template <class FinalPipeline, class FinalPipelineState, class FrgTensorC, class ClusterMask>
  CUTLASS_DEVICE void
  mma(MainloopPipeline& pipeline, PipelineState& slm_pipe_read, MainloopPipelineB& pipeline_b, PipelineStateB& slm_pipe_read_b, FinalPipeline finalPipeline, FinalPipelineState& finalPipelineState, FrgTensorC& accumulator, int k_tile_count, int local_id, ClusterMask const& cluster_mask, TensorStorage& shared_tensors) {
    static_assert(cute::rank(SmemLayoutA{}) == 3, "Smem layout must be rank 3.");
    static_assert(cute::rank(SmemLayoutB{}) == 3, "Smem layout must be rank 3.");
    static_assert(cute::is_void_v<SmemCopyAtomA>,
      "XE4 GMMA mainloops cannot have a non-void copy atom for smem sourced instructions.");
    static_assert(cute::is_void_v<SmemCopyAtomB>,
      "XE4 GMMA mainloops cannot have a non-void copy atom for smem sourced instructions.");

    auto sA = make_tensor(shared_tensors.smem_A.begin(), SmemLayoutA {});
    auto sB = make_tensor(shared_tensors.smem_B.begin(), SmemLayoutB {});
    auto sMetaA = make_tensor(shared_tensors.smem_metaA.data(), SmemLayoutMetaA {});
    auto sMetaB = make_tensor(shared_tensors.smem_metaB.data(), SmemLayoutMetaB {});

    auto [cluster_mask_a, cluster_mask_b] = cluster_mask;

    TiledMma tiled_mma;
    constexpr auto dstType = C<cute::xe4::GMMA::DstType::Accum>{};
    constexpr auto scaleOutOne = C<cute::xe4::GMMA::ScaleOut::One>{};
    constexpr auto scaleOutZero = C<cute::xe4::GMMA::ScaleOut::Zero>{};

    auto thread_mma = tiled_mma.get_thread_slice(0);
    auto tCsA = thread_mma.partition_fragment_A(sA);            // (MMA,MMA_M,MMA_K,PIPE)
    auto tCsB = thread_mma.partition_fragment_B(sB);            // (MMA,MMA_N,MMA_K,PIPE)
    auto accum_ = thread_mma.partition_fragment_C(accumulator);  // (MMA,MMA_M,MMA_N)
    auto accum = make_tensor(accum_.data(), append(accum_.layout(), Layout<_1,_0>{}));
    auto tEsMetaA = TiledMma::make_fragment_E(partition_E(thread_mma, sMetaA, TVLayoutMetaA{}));    // (TMA,TMA_M,TMA_K,PIPE)
    auto tEsMetaB = TiledMma::make_fragment_E(partition_E(thread_mma, sMetaB, TVLayoutMetaB{}));    // (TMA,TMA_N,TMA_K,PIPE)

    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(accum));                // M
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));                 // K
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA));   // PIPE
    CUTE_STATIC_ASSERT_V(Int<StagesB>{} == size<2>(sB));                  // PIPE

    auto cshape = ClusterShape{};
    auto wg_expect_tx = size<1>(accum) * size<2>(accum) * size<2>(tCsA);
    auto cluster_expect_tx = wg_expect_tx * (size<0>(cshape) + size<1>(cshape));

    pipeline.consumer_try_wait(slm_pipe_read);
    uint32_t read_stage = slm_pipe_read.index();
    auto abar_cons = pipeline.consumer_get_barrier(slm_pipe_read);

    for (int ib = 0; ib < SplitB; ++ib, ++slm_pipe_read_b) {
      uint32_t read_stage_b = slm_pipe_read_b.index();
      pipeline_b.consumer_try_wait(slm_pipe_read_b);
      auto abar_cons_b = pipeline_b.consumer_get_barrier(slm_pipe_read_b);

      cute::gemm(tiled_mma.with(scaleOutZero, dstType, abar_cons, cluster_mask_a, abar_cons_b, cluster_mask_b),
        make_zip_tensor(tCsA(_,_,0,read_stage), tEsMetaA(_,_,0,read_stage)),
        make_zip_tensor(tCsB(_,_,0,read_stage_b), tEsMetaB(_,_,0,read_stage_b)), accum(_,_,ib,_));

      for (int k_block = 1; k_block < size<2>(tCsA); ++k_block) {
        cute::gemm(tiled_mma.with(scaleOutOne, dstType, abar_cons, cluster_mask_a, abar_cons_b, cluster_mask_b),
          make_zip_tensor(tCsA(_,_,k_block,read_stage), tEsMetaA(_,_,k_block,read_stage)),
          make_zip_tensor(tCsB(_,_,k_block,read_stage_b), tEsMetaB(_,_,k_block,read_stage_b)), accum(_,_,ib,_));
      }

      pipeline_b.consumer_commit(slm_pipe_read_b, 1);
    }
    pipeline.consumer_commit(slm_pipe_read, SplitB);
    ++slm_pipe_read;

    for (uint32_t i = 1; i < k_tile_count-1; ++i, ++slm_pipe_read) {
      uint32_t read_stage = slm_pipe_read.index();
      pipeline.consumer_try_wait(slm_pipe_read);
      auto abar_cons = pipeline.consumer_get_barrier(slm_pipe_read);

      for (int ib = 0; ib < SplitB; ++ib, ++slm_pipe_read_b) {
        uint32_t read_stage_b = slm_pipe_read_b.index();
        pipeline_b.consumer_try_wait(slm_pipe_read_b);
        auto abar_cons_b = pipeline_b.consumer_get_barrier(slm_pipe_read_b);
        cute::gemm(tiled_mma.with(scaleOutOne, dstType, abar_cons, cluster_mask_a, abar_cons_b, cluster_mask_b),
        make_zip_tensor(tCsA(_,_,_,read_stage), tEsMetaA(_,_,_,read_stage)),
        make_zip_tensor(tCsB(_,_,_,read_stage_b), tEsMetaB(_,_,_,read_stage_b)), accum(_,_,ib,_));
        pipeline_b.consumer_commit(slm_pipe_read_b, 1);
      }
      pipeline.consumer_commit(slm_pipe_read, SplitB);
    }

    {
      uint32_t read_stage = slm_pipe_read.index();
      pipeline.consumer_try_wait(slm_pipe_read);
      auto abar_cons = pipeline.consumer_get_barrier(slm_pipe_read);
      auto abar_cons_d = finalPipeline.store_get_barrier(finalPipelineState);

      for (int ib = 0; ib < SplitB; ++ib, ++slm_pipe_read_b) {
        uint32_t read_stage_b = slm_pipe_read_b.index();
        pipeline_b.consumer_try_wait(slm_pipe_read_b);
        auto abar_cons_b = pipeline_b.consumer_get_barrier(slm_pipe_read_b);
        cute::gemm(tiled_mma.with(scaleOutOne, dstType, abar_cons_d, abar_cons, cluster_mask_a, abar_cons_b, cluster_mask_b),
          make_zip_tensor(tCsA(_,_,_,read_stage), tEsMetaA(_,_,_,read_stage)),
          make_zip_tensor(tCsB(_,_,_,read_stage_b), tEsMetaB(_,_,_,read_stage_b)), accum(_,_,ib,_));
        pipeline_b.consumer_commit(slm_pipe_read_b, 1);
      }
      pipeline.consumer_commit(slm_pipe_read, SplitB);
      finalPipeline.store_commit(finalPipelineState, SplitB);
      finalPipeline.store_try_wait(finalPipelineState);
      ++finalPipelineState;
    }
  }

public:
  template <class MMA_Atom,
            class AtomLayoutMNK,
            class PermutationMNK,
            class ETensor,
            class AtomLayoutE_TV>
  CUTE_HOST_DEVICE static constexpr
  auto
  thrfrg_E(TiledMMA<MMA_Atom, AtomLayoutMNK, PermutationMNK> const& mma, ETensor&& etensor, AtomLayoutE_TV const& atom_layoute_tv)
  {
    using TiledMma = TiledMMA<MMA_Atom, AtomLayoutMNK, PermutationMNK>;

    CUTE_STATIC_ASSERT_V(rank(etensor) >= Int<2>{});

    // Reorder the tensor for the TiledAtom
    auto t_tile = make_tile(get<0>(PermutationMNK{}),
                            get<2>(PermutationMNK{}));
    auto t_tensor = logical_divide(etensor, t_tile);                 // (PermM,PermK)

    // Tile the tensor for the Atom
    auto e_tile = make_tile(make_layout(size<0>(ScaleTileShape{})),
                            make_layout(size<2>(ScaleTileShape{})));
    auto e_tensor = zipped_divide(t_tensor, e_tile);                 // ((AtomM,AtomK),(RestM,RestK))

    // Transform the Atom mode from (M,K) to (Thr,Val)
    auto tv_tensor = e_tensor.compose(AtomLayoutE_TV{},_);           // ((ThrV,FrgV),(RestM,RestK))

    // Tile the tensor for the Thread
    auto thr_tile = make_tile(_,
                              make_tile(make_layout(size<1>(mma.thr_layout_vmnk_)),
                                        make_layout(size<3>(mma.thr_layout_vmnk_))));
    auto thr_tensor = zipped_divide(tv_tensor, thr_tile);            // ((ThrV,(ThrM,ThrK)),(FrgV,(RestM,RestK)))

    return thr_tensor;
  }

  template<class... MArgs,
          class AtomLayoutE_TV>
  CUTE_HOST_DEVICE static constexpr
  auto
  get_layoutE_TV(TiledMMA<MArgs...> const& mma, AtomLayoutE_TV const& atom_layoute_tv)
  {
    // (M,K) -> (M,K)
    auto ref_E = make_layout(make_shape(tile_size<0>(mma), tile_size<2>(mma)));
    // (ethrid,val) -> (M,K)
    auto layoutE_TV = thrfrg_E(mma, ref_E, atom_layoute_tv);

    // (ThrV,(ThrM,ThrK)) -> (ThrV,(ThrM,ThrN,ThrK))
    auto etile = make_tile(_,
                            make_tile(make_layout(make_shape (size<1>(mma.thr_layout_vmnk_), size<2>(mma.thr_layout_vmnk_)),
                                                  make_stride(               Int<1>{} ,                Int<0>{} )),
                                      _));

    // thr_idx -> (ThrV,ThrM,ThrN,ThrK)
    auto thridx_2_thrid = right_inverse(mma.thr_layout_vmnk_);

    // (thr_idx,val) -> (M,K)
    return layoutE_TV.compose(etile, _).compose(thridx_2_thrid, _);
  }

  template <class... MArgs, class ETensor, class AtomLayoutE_TV>
  CUTE_HOST_DEVICE static constexpr
  auto
  partition_E(ThrMMA<MArgs...> const& thr_mma, ETensor&& etensor, AtomLayoutE_TV const& atom_layoute_tv)
  {
    auto thr_tensor = make_tensor(static_cast<ETensor&&>(etensor).data(), thrfrg_E(thr_mma, etensor.layout(), atom_layoute_tv));

    auto thr_vmk = make_coord(get<0>(thr_mma.thr_vmnk_), make_coord(get<1>(thr_mma.thr_vmnk_), get<3>(thr_mma.thr_vmnk_)));
    return thr_tensor(thr_vmk, make_coord(_, repeat<rank<1,1>(thr_tensor)>(_)));
  }

  template <class... CArgs, class... MArgs, class AtomLayoutE_TV, class Tiler>
  CUTE_HOST_DEVICE static constexpr
  auto
  make_tiled_copy_E(Copy_Atom<CArgs...> const& copy_atom,
                    TiledMMA<MArgs...>  const& mma,
                    AtomLayoutE_TV const& atom_layoute_tv,
                    Tiler const& tiler)
  {
    return make_tiled_copy_impl(copy_atom, get_layoutE_TV(mma, atom_layoute_tv), tiler);
  }
};

}