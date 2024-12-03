#pragma once

#include "cute/atom/mma_traits_xe4_amma.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/gemm/collective/builders/xe4_sparse_config.inl"

namespace cutlass::gemm::collective {

using namespace cute;
using namespace cute::detail;
using namespace cutlass::gemm;

template <
  int Stages,
  uint32_t SplitB,
  class ClusterShape,
  class KernelSchedule,
  class TileShape_,
  class ElementA_,
  class LayoutPairAE_,
  class ElementB_,
  class LayoutPairBE_,
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
  MainloopXe4DmaGmmaWarpSpecializedScale<Stages, SplitB, ClusterShape, KernelSchedule>,
  TileShape_,
  ElementA_,
  LayoutPairAE_,
  ElementB_,
  LayoutPairBE_,
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
  using DispatchPolicy = MainloopXe4DmaGmmaWarpSpecializedScale<Stages, SplitB, ClusterShape, KernelSchedule>;
  using TileShape = TileShape_;
  using TiledMma = TiledMma_;
  using ElementA = ElementA_;
  using ElementAMma = typename TiledMma::ValTypeA;
  using ElementAMmaRaw = typename ElementAMma::raw_type;
  using LayoutPairAE = LayoutPairAE_;
  using LayoutA = remove_cvref_t<decltype(get<0>(LayoutPairAE{}))>;
  using LayoutMetaA = remove_cvref_t<decltype(get<1>(LayoutPairAE{}))>;
  using StrideA = decltype(cute::stride(LayoutA{}));
  using ElementB = ElementB_;
  using ElementBMma = typename TiledMma::ValTypeB;
  using ElementBMmaRaw = typename ElementBMma::raw_type;
  using LayoutPairBE = LayoutPairBE_;
  using LayoutB = remove_cvref_t<decltype(get<0>(LayoutPairBE{}))>;
  using LayoutMetaB = remove_cvref_t<decltype(get<1>(LayoutPairBE{}))>;
  using StrideB = decltype(cute::stride(LayoutB{}));
  using ElementEMma = typename TiledMma::ValTypeE;
  using ElementE = typename ElementEMma::raw_type;
  using GmemTiledCopyA = GmemTiledCopyA_;
  using GmemTiledCopyB = GmemTiledCopyB_;
  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;
  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB = SmemCopyAtomB_;
  using TileShapeMma = typename TiledMma::Shape_MNK;

  using TensorDescPtr = uint64_t*;
  using AbarrierPtr = uint64_t*;
  using MatrixDesc = uint32_t;

  using ElementC = typename TiledMma::ValTypeD;
  using ElementAccumulator = typename TiledMma::ValTypeC;
  using StrideC = cutlass::detail::TagToStrideC_t<cutlass::layout::RowMajor>;

  static_assert(is_sparse<ElementAMma>::value, "ElementAMma is sparse");
  static_assert(is_sparse<ElementBMma>::value, "ElementBMma is sparse");
  static_assert(is_sparse<ElementEMma>::value, "ElementEMma is sparse");

  static constexpr cute::xe4::GMMA::Major tnspA = TiledMma::tnspA;
  static constexpr cute::xe4::GMMA::Major tnspB = TiledMma::tnspB;
  static constexpr cute::xe4::GMMA::Major tnspE = cute::xe4::GMMA::Major::MN;
  static constexpr int ElementAMmaSparsity = ElementAMma::sparsity;
  static constexpr int ElementBMmaSparsity = ElementBMma::sparsity;
  static constexpr int ElementEMmaSparsity = ElementEMma::sparsity;

  using SparseConfig = cutlass::Xe4GemmSparseConfig<ElementAMma, tnspA, ElementBMma, tnspB, ElementEMma>;

  // Metadata pathways
  using SmemCopyAtomE = AutoVectorizingCopy;

  using AuxParamsA = AuxParams<(tnspA == cute::xe4::GMMA::Major::K ? slm_matrix_type::type1 : slm_matrix_type::type2), tnspA, TensorDescPtr, 0>;
  using AuxParamsB = AuxParams<slm_matrix_type::type1, tnspB, TensorDescPtr, 1>;
  using AuxParamsMetaA = AuxParams<slm_matrix_type::type2, tnspE, TensorDescPtr, 3, true>;
  using AuxParamsMetaB = AuxParams<slm_matrix_type::type2, tnspE, TensorDescPtr, 4, true>;

  using MainloopPipeline = cutlass::xe4::PipelineTmaAsync<Stages, 0, AbarrierPtr>;
  using PipelineState = cutlass::xe4::PipelineState<Stages>;

  static constexpr uint32_t StagesB = (Stages - 1) * SplitB + 1;
  using MainloopPipelineB = cutlass::xe4::PipelineTmaAsync<StagesB, 1, AbarrierPtr>;
  using PipelineStateB = typename MainloopPipelineB::PipelineState;

  static_assert(DispatchPolicy::Stages >= 2, "Specialization requires Stages set to value 2 or more.");
  static_assert(cute::is_same_v<GmemTiledCopyA, cute::xe4::ASYNC_TENSOR_LOAD> || cute::is_same_v<GmemTiledCopyA, cute::xe4::ASYNC_TENSOR_LOAD_MULTICAST>,
      "GmemTiledCopy - invalid XE4 DMA copy atom specified.");
  static_assert(cute::is_same_v<GmemTiledCopyB, cute::xe4::ASYNC_TENSOR_LOAD> || cute::is_same_v<GmemTiledCopyB, cute::xe4::ASYNC_TENSOR_LOAD_MULTICAST>,
      "GmemTiledCopy - invalid XE4 DMA copy atom specified.");

  static_assert(cute::rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M,K)");
  static_assert((size<0>(TileShapeMma{}) % size<0>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShapeMma{}) % size<1>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(cute::rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (N,K)");
  static_assert((size<1>(TileShapeMma{}) % size<0>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShapeMma{}) % size<1>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static constexpr auto SplitTileShape = replace<1>(TileShapeMma{}, shape<1>(TileShapeMma{})/C<SplitB>{});

  // Tile along modes in a way that maximizes the TMA box size.
  using SmemLayoutA = decltype(tile_to_shape(
    SmemLayoutAtomA{},
    make_shape(shape<0>(TileShapeMma{}), shape<2>(TileShapeMma{}), Int<DispatchPolicy::Stages>{}),
    cute::conditional_t<tnspA == cute::xe4::GMMA::Major::K, Step<_2,_1,_3>, Step<_1,_2,_3>>{}));

  using SmemLayoutB = decltype(tile_to_shape(
    SmemLayoutAtomB{},
    make_shape(shape<1>(TileShapeMma{}), shape<2>(TileShapeMma{}), Int<StagesB>{}),
    cute::conditional_t<tnspB == cute::xe4::GMMA::Major::K, Step<_2,_1,_3>, Step<_1,_2,_3>>{}));

  using MetaTileShape = decltype(SparseConfig::template deduce_MetaTileShape<TileShapeMma>());
  using MetaALayout = cute::xe4::ABLayout<get<0>(MetaTileShape{}), get<2>(MetaTileShape{})>;
  using MetaBLayout = cute::xe4::ABLayout<get<1>(MetaTileShape{}), get<2>(MetaTileShape{})>;

  // MetaA is a column-major WgM x MetaK matrix
  using SmemLayoutMetaA = decltype(tile_to_shape(
    make_layout(Shape<_32, _32>{}, GenColMajor{}),
    make_shape(shape<0>(MetaTileShape{}), shape<2>(MetaTileShape{}), Int<DispatchPolicy::Stages>{})));

  // MetaB is a column-major WgN x MetaK matrix
  using SmemLayoutMetaB = decltype(tile_to_shape(
    make_layout(Shape<_32, _32>{}, GenColMajor{}),
    make_shape(shape<1>(MetaTileShape{}), shape<2>(MetaTileShape{}), Int<StagesB>{})));

  using TmaInternalTypeA = uint_bit_t<sizeof_bits_v<ElementAMmaRaw>>;
  using TmaInternalTypeB = uint_bit_t<sizeof_bits_v<ElementBMmaRaw>>;
  using TmaInternalTypeE = uint8_t;

  struct SharedStorage
  {
    struct TensorStorage
    {
      cute::ArrayEngine<ElementAMma, cute::cosize_v<SmemLayoutA>> smem_A;
      cute::ArrayEngine<ElementBMma, cute::cosize_v<SmemLayoutB>> smem_B;
      cute::array<uint8_t, cute::cosize_v<SmemLayoutMetaA>> smem_metaA;
      cute::array<uint8_t, cute::cosize_v<SmemLayoutMetaB>> smem_metaB;
    };
  };

  using TensorStorage = typename SharedStorage::TensorStorage;

  static constexpr auto wgMetaK = SparseConfig::deduce_wgMetaK(get<2>(TileShapeMma{}));
  static constexpr auto wgMetaKStep = uint32_t(SparseConfig::deduce_wgMetaKStep(get<2>(TileShapeMma{})));

  static constexpr uint32_t TransactionBytes_A =
        cutlass::bits_to_bytes(cosize(take<0,2>(SmemLayoutA{})) * cute::sizeof_bits_v<ElementAMma>) +
        uint32_t(size<0>(SmemLayoutMetaA{})*wgMetaK) * sizeof(ElementE);

  static constexpr uint32_t TransactionBytes_B =
        cutlass::bits_to_bytes(cosize(take<0,2>(SmemLayoutB{})) * cute::sizeof_bits_v<ElementBMma>) +
        uint32_t(size<0>(SmemLayoutMetaB{})*wgMetaK) * sizeof(ElementE);

  // Host side kernel arguments
  struct Arguments {
    ElementA const* ptr_A {nullptr};
    LayoutA layout_a;
    ElementB const* ptr_B {nullptr};
    LayoutB layout_b;
    ElementE const* ptr_metaA {nullptr};
    LayoutMetaA layout_meta_a;
    ElementE const* ptr_metaB {nullptr};
    LayoutMetaB layout_meta_b;
  };

  // Device side kernel params
  struct Params {
    using TiledLoadA = decltype(make_xe4_copy<GmemTiledCopyA, AuxParamsA, TmaInternalTypeA>(
      make_tensor(recast_ptr<ElementAMma>(nullptr), LayoutA{}),
      SmemLayoutA{}(_, _, _0{}), make_shape(shape<0>(TileShapeMma{}), shape<2>(TileShapeMma{})), size<1>(ClusterShape{})));

    using TiledLoadB = decltype(make_xe4_copy<GmemTiledCopyB, AuxParamsB, TmaInternalTypeB>(
      make_tensor(recast_ptr<ElementBMma>(nullptr), LayoutB{}),
      SmemLayoutB{}(_, _, _0{}), make_shape(shape<1>(TileShapeMma{}), shape<2>(TileShapeMma{})), size<0>(ClusterShape{})));

    using TiledLoadMetaA = decltype(make_xe4_copy<GmemTiledCopyA, AuxParamsMetaA, TmaInternalTypeE>(
      make_tensor(recast_ptr<ElementEMma>(nullptr), LayoutMetaA{}),
      SmemLayoutA{}(_, _, _0{}), make_shape(shape<0>(TileShapeMma{}), shape<2>(TileShapeMma{})), size<1>(ClusterShape{})));

    using TiledLoadMetaB = decltype(make_xe4_copy<GmemTiledCopyB, AuxParamsMetaB, TmaInternalTypeE>(
      make_tensor(recast_ptr<ElementEMma>(nullptr), LayoutMetaB{}),
      SmemLayoutB{}(_, _, _0{}), make_shape(shape<1>(TileShapeMma{}), shape<2>(TileShapeMma{})), size<0>(ClusterShape{})));

    TiledLoadA load_a;
    TiledLoadB load_b;
    TiledLoadMetaA load_meta_a;
    TiledLoadMetaB load_meta_b;

    LayoutA layout_a;
    LayoutB layout_b;
    LayoutMetaA layout_meta_a;
    LayoutMetaB layout_meta_b;
  };

  template<class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args) {
    // Optionally append 1s until problem shape is rank-4 (MNKL), in case it is only rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M, N, K, L] = problem_shape;

    auto ptr_A = recast_ptr<ElementAMma>(args.ptr_A);
    auto ptr_B = recast_ptr<ElementBMma>(args.ptr_B);
    auto ptr_metaA = recast_ptr<ElementEMma>(args.ptr_metaA);
    auto ptr_metaB = recast_ptr<ElementEMma>(args.ptr_metaB);

    Tensor tensor_a = make_tensor(ptr_A, args.layout_a);
    Tensor tensor_b = make_tensor(ptr_B, args.layout_b);
    Tensor tensor_meta_a = make_tensor(ptr_metaA, args.layout_meta_a);
    Tensor tensor_meta_b = make_tensor(ptr_metaB, args.layout_meta_b);

    auto load_a = make_xe4_copy<GmemTiledCopyA, AuxParamsA, TmaInternalTypeA>(
      tensor_a,
      SmemLayoutA{}(_, _, _0{}),
      make_shape(shape<0>(TileShapeMma{}), shape<2>(TileShapeMma{})),
      size<1>(ClusterShape{}));   // mcast along N mode for this M load, if any

    auto load_b = make_xe4_copy<GmemTiledCopyB, AuxParamsB, TmaInternalTypeB>(
      tensor_b,
      SmemLayoutB{}(_, _, _0{}),
      make_shape(shape<1>(TileShapeMma{}), shape<2>(TileShapeMma{})),
      size<0>(ClusterShape{}));   // mcast along M mode for this N load, if any

    auto load_meta_a = make_xe4_copy<GmemTiledCopyA, AuxParamsMetaA, TmaInternalTypeE>(
      tensor_meta_a,
      SmemLayoutA{}(_, _, _0{}),
      make_shape(shape<0>(TileShapeMma{}), shape<2>(TileShapeMma{})),
      size<1>(ClusterShape{})); // mcast along N mode for this M load, if any

    auto load_meta_b = make_xe4_copy<GmemTiledCopyB, AuxParamsMetaB, TmaInternalTypeE>(
      tensor_meta_b,
      SmemLayoutB{}(_, _, _0{}),
      make_shape(shape<1>(TileShapeMma{}), shape<2>(TileShapeMma{})),
      size<0>(ClusterShape{}));   // mcast along M mode for this N load, if any

    return {load_a, load_b, load_meta_a, load_meta_b, args.layout_a, args.layout_b, args.layout_meta_a, args.layout_meta_b};
  }

  template <class ProblemShape>
  CUTLASS_DEVICE auto
  load_init(ProblemShape const& problem_shape, Params const& mainloop_params) const {
    auto [M, N, K, L] = problem_shape;

    auto mA_mkl = mainloop_params.load_a.get_tma_tensor(mainloop_params.layout_a.shape());   // (m,k,l)
    auto mB_nkl = mainloop_params.load_b.get_tma_tensor(mainloop_params.layout_b.shape());   // (n,k,l)

    auto mMetaA_mkl = mainloop_params.load_meta_a.get_tma_tensor(mainloop_params.layout_meta_a.shape());   // (m,k,l)
    auto mMetaB_nkl = mainloop_params.load_meta_b.get_tma_tensor(mainloop_params.layout_meta_b.shape());   // (n,k,l)

    auto gA_mkl = local_tile(mA_mkl, TileShape_{}, make_coord(_,_,_), Step<_1, X,_1>{});        // (BLK_M,BLK_K,m,k,l)
    auto gB_nkl = local_tile(mB_nkl, TileShape_{}, make_coord(_,_,_), Step< X,_1,_1>{});        // (BLK_N,BLK_K,n,k,l)

    auto gMetaA_mkl = local_tile(mMetaA_mkl, TileShape_{}, make_coord(_,_,_), Step<_1, X,_1>{});   // (BLK_M,META_K,m,k,l)
    auto gMetaB_nkl = local_tile(mMetaB_nkl, TileShape_{}, make_coord(_,_,_), Step< X,_1,_1>{});   // (BLK_N,META_K,n,k,l)

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

    auto gMetaB = gMetaB_nkl(_, _, n_coord, _, l_coord);      // (BLK_N,BLK_K,k)
    auto tBgMetaB = block_load_meta_b.partition_S(gMetaB);    // (TMA,TMA_N,TMA_K,k)

    TiledMma tiled_mma;
    auto copy_atom_E = Copy_Atom<SmemCopyAtomE, uint32_t>{};

    auto metaA_tiler = make_shape(get<0>(MetaTileShape{}), get<2>(MetaTileShape{}));
    auto smem_tiled_copy_MetaA = make_tiled_copy_E(copy_atom_E, tiled_mma, MetaALayout{}, metaA_tiler);
    auto smem_thr_copy_MetaA   = smem_tiled_copy_MetaA.get_thread_slice(0);
    auto tEsMetaA = smem_thr_copy_MetaA.partition_D(sMetaA);    // (TMA,TMA_M,TMA_K,PIPE)

    auto metaB_tiler = make_shape(get<1>(MetaTileShape{}), get<2>(MetaTileShape{}));
    auto smem_tiled_copy_MetaB = make_tiled_copy_E(copy_atom_E, tiled_mma, MetaBLayout{}, metaB_tiler);
    auto smem_thr_copy_MetaB   = smem_tiled_copy_MetaB.get_thread_slice(0);
    auto tEsMetaB = smem_thr_copy_MetaB.partition_D(sMetaB);    // (TMA,TMA_N,TMA_K,PIPE)

    for (int i = 0; i < k_tile_count; ++i, ++slm_pipe_write) {
      pipeline.producer_try_wait(slm_pipe_write);

      uint32_t write_stage = slm_pipe_write.index();
      auto abar_prod = pipeline.producer_get_barrier(slm_pipe_write);

      constexpr auto dimIndex = _2{};
      const uint32_t newDimSize = (i+1) * wgMetaKStep;  // set dim size to make dma load meta with OOB
      copy(mainloop_params.load_a.with(abar_prod, cluster_mask_a), tAgA(_,_,_,i), tAsA(_,_,_,write_stage));
      copy(mainloop_params.load_meta_a.with(dimIndex, newDimSize, abar_prod, cluster_mask_a), tAgMetaA(_,_,_,i), tEsMetaA(_,_,_,write_stage));
      pipeline.producer_commit(slm_pipe_write, TransactionBytes_A);

      for (int bIdx = 0; bIdx < SplitB; ++bIdx, ++slm_pipe_write_b) {
        pipeline_b.producer_try_wait(slm_pipe_write_b);
        uint32_t write_stage_b = slm_pipe_write_b.index();
        auto abar_prod_b = pipeline_b.producer_get_barrier(slm_pipe_write_b);
        copy(mainloop_params.load_b.with(abar_prod_b, cluster_mask_b), tBgB(_,bIdx,_,i), tBsB(_,_0{},_,write_stage_b));
        copy(mainloop_params.load_meta_b.with(dimIndex, newDimSize, abar_prod_b, cluster_mask_b), tBgMetaB(_,bIdx,_,i), tEsMetaB(_,_0{},_,write_stage_b));
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
    auto tEsMetaA = TiledMma::make_fragment_E(partition_E(thread_mma, sMetaA, MetaALayout{}));    // (TMA,TMA_M,TMA_K,PIPE)
    auto tEsMetaB = TiledMma::make_fragment_E(partition_E(thread_mma, sMetaB, MetaBLayout{}));    // (TMA,TMA_N,TMA_K,PIPE)

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
    auto e_tile = make_tile(make_layout(size<0>(MetaTileShape{})),
                            make_layout(size<2>(MetaTileShape{})));
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