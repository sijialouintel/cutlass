#pragma once

#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/pipeline/pipeline.hpp"

#include "cutlass/util/packed_stride.hpp"
#include "cute/atom/mma_traits_xe4_amma.hpp"

namespace cutlass::gemm::collective {

using namespace cute;
using namespace cute::detail;
using namespace cutlass::gemm;

template <
  int Stages,
  class ClusterShape,
  class KernelSchedule,
  class TileShape_,
  class ElementA_,
  class StrideA_,
  class ElementB_,
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
  MainloopXe4DmaGmmaWarpSpecialized<Stages, ClusterShape, KernelSchedule>,
  TileShape_,
  ElementA_,
  StrideA_,
  ElementB_,
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
  using DispatchPolicy = MainloopXe4DmaGmmaWarpSpecialized<Stages, ClusterShape, KernelSchedule>;
  using TileShape = TileShape_;
  using ElementA = ElementA_;
  using StrideA = StrideA_;
  using ElementB = ElementB_;
  using StrideB = StrideB_;
  using TiledMma = TiledMma_;
  using GmemTiledCopyA = GmemTiledCopyA_;
  using GmemTiledCopyB = GmemTiledCopyB_;
  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;
  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB = SmemCopyAtomB_;

  using TensorDescPtr = uint64_t*;
  using AbarrierPtr = uint64_t*;
  using MatrixDesc = uint32_t;

  using ElementC = typename TiledMma::ValTypeD;
  using ElementAccumulator = typename TiledMma::ValTypeC;
  using StrideC = cutlass::detail::TagToStrideC_t<cutlass::layout::RowMajor>;

  static constexpr bool IsRowMajorA = cutlass::detail::is_major<1, StrideA>();
  static constexpr bool IsRowMajorB = cutlass::detail::is_major<0, StrideB>();

  using AuxParamsA = AuxParams<
    (IsRowMajorA ? slm_matrix_type::type1 : slm_matrix_type::type2),
    TensorDescPtr,
    0
  >;
  using AuxParamsB = AuxParams<slm_matrix_type::type1, TensorDescPtr, 1>;

  using MainloopPipeline = cutlass::xe4::PipelineTmaAsync<Stages, 0, AbarrierPtr>;
  using PipelineState = cutlass::xe4::PipelineState<Stages>;

  static_assert(DispatchPolicy::Stages >= 2, "Specialization requires Stages set to value 2 or more.");
  static_assert(cute::is_same_v<GmemTiledCopyA, cute::xe4::ASYNC_TENSOR_LOAD> || cute::is_same_v<GmemTiledCopyA, cute::xe4::ASYNC_TENSOR_LOAD_MULTICAST>,
      "GmemTiledCopy - invalid XE4 DMA copy atom specified.");
  static_assert(cute::is_same_v<GmemTiledCopyB, cute::xe4::ASYNC_TENSOR_LOAD> || cute::is_same_v<GmemTiledCopyB, cute::xe4::ASYNC_TENSOR_LOAD_MULTICAST>,
      "GmemTiledCopy - invalid XE4 DMA copy atom specified.");

  // Tile along modes in a way that maximizes the TMA box size.
  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{},
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{}),
      cute::conditional_t<IsRowMajorA, Step<_2,_1,_3>, Step<_1,_2,_3>>{}));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{}),
      cute::conditional_t<!IsRowMajorB, Step<_2,_1,_3>, Step<_1,_2,_3>>{}));

  struct SharedStorage
  {
    struct TensorStorage
    {
      cute::array<ElementA, cute::cosize_v<SmemLayoutA>> smem_A;
      cute::array<ElementB, cute::cosize_v<SmemLayoutB>> smem_B;
    };
  };

  using TensorStorage = typename SharedStorage::TensorStorage;

  // Host side kernel arguments
  struct Arguments {
    ElementA const* ptr_A {nullptr};
    StrideA dA;
    ElementB const* ptr_B {nullptr};
    StrideB dB;
  };

  // Device side kernel params
  struct Params {
    using TiledLoadA = decltype(make_xe4_copy<GmemTiledCopyA, AuxParamsA>(
      make_tensor(static_cast<ElementA const*>(nullptr), repeat_like(StrideA{}, int32_t(0)), StrideA{}),
      SmemLayoutA{}(_, _, _0{}), make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})), size<1>(ClusterShape{})));

    using TiledLoadB = decltype(make_xe4_copy<GmemTiledCopyB, AuxParamsB>(
      make_tensor(static_cast<ElementB const*>(nullptr), repeat_like(StrideB{}, int32_t(0)), StrideB{}),
      SmemLayoutB{}(_, _, _0{}), make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})), size<0>(ClusterShape{})));

    TiledLoadA load_a;
    TiledLoadB load_b;
  };

  template<class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args) {
    auto [M, N, K, L] = problem_shape;

    auto A = make_tensor(args.ptr_A, make_layout(make_shape(M,K,L), args.dA));
    auto B = make_tensor(args.ptr_B, make_layout(make_shape(N,K,L), args.dB));

    auto load_a = make_xe4_copy<GmemTiledCopyA, AuxParamsA>(A, SmemLayoutA{}(_, _, _0{}), make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})), size<1>(ClusterShape{}));
    auto load_b = make_xe4_copy<GmemTiledCopyB, AuxParamsB>(B, SmemLayoutB{}(_, _, _0{}), make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})), size<0>(ClusterShape{}));

    return {load_a, load_b};
  }

  template <class ProblemShape>
  CUTLASS_DEVICE auto
  load_init(ProblemShape const& problem_shape, Params const& mainloop_params) const {
    auto [M, N, K, L] = problem_shape;

    auto mA_mkl = mainloop_params.load_a.get_tma_tensor(make_shape(M, K, L));   // (m,k,l)
    auto mB_knl = mainloop_params.load_b.get_tma_tensor(make_shape(N, K, L));   // (n,k,l)

    auto gA_mkl = flat_divide(mA_mkl, make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})));  // (BLK_M,BLK_K,m,k,l)
    auto gB_knl = flat_divide(mB_knl, make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})));  // (BLK_N,BLK_K,n,k,l)

    return cute::make_tuple(gA_mkl, gB_knl);
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

  template <class TensorA, class TensorB, class BlockCoord, class ClusterMask>
  CUTLASS_DEVICE void
  load(Params const& mainloop_params, MainloopPipeline pipeline, PipelineState slm_pipe_write,
    cute::tuple<TensorA, TensorB> const& load_inputs, BlockCoord const& blk_coord, int k_tile_count, int local_id, ClusterMask const& cluster_mask, TensorStorage& shared_tensors) {
    auto sA = make_tensor(reinterpret_cast<ElementA *>(shared_tensors.smem_A.data()), SmemLayoutA {});
    auto sB = make_tensor(reinterpret_cast<ElementB *>(shared_tensors.smem_B.data()), SmemLayoutB {});

    auto [load_a, load_b] = mainloop_params;
    auto [cluster_mask_a, cluster_mask_b] = cluster_mask;

    uint32_t cluster_wgid_x = get_cluster_wgid<0>();
    uint32_t cluster_wgid_y = get_cluster_wgid<1>();

    auto block_load_a = load_a.get_slice(cluster_wgid_x);
    auto block_load_b = load_b.get_slice(cluster_wgid_y);

    auto [gA_mkl, gB_knl] = load_inputs;
    auto [m_coord, n_coord, l_coord] = blk_coord;

    auto gA = gA_mkl(_, _, m_coord, _, l_coord);        // (BLK_M,BLK_K,k)
    auto tAgA = block_load_a.partition_S(gA);           // (TMA,TMA_M,TMA_K,k)
    auto tAsA = block_load_a.partition_D(sA);           // (TMA,TMA_M,TMA_K,PIPE)

    auto gB = gB_knl(_, _, n_coord, _, l_coord);        // (BLK_N,BLK_K,k)
    auto tBgB = block_load_b.partition_S(gB);           // (TMA,TMA_N,TMA_K,k)
    auto tBsB = block_load_b.partition_D(sB);           // (TMA,TMA_N,TMA_K,PIPE)

    constexpr uint32_t slm_bytes_load = (sizeof(TensorStorage::smem_A) + sizeof(TensorStorage::smem_B)) / Stages;

    for (int i = 0; i < k_tile_count; ++i, ++slm_pipe_write) {
      pipeline.producer_try_wait(slm_pipe_write);

      uint32_t write_stage = slm_pipe_write.index();
      auto abar_prod = pipeline.producer_get_barrier(slm_pipe_write);

      copy(load_a.with(abar_prod, cluster_mask_a), tAgA(_,_,_,i), tAsA(_,_,_,write_stage));
      copy(load_b.with(abar_prod, cluster_mask_b), tBgB(_,_,_,i), tBsB(_,_,_,write_stage));

      pipeline.producer_commit(slm_pipe_write, slm_bytes_load);
    }
  }

  template <class FinalPipeline, class FinalPipelineState, class FrgTensorC, class ClusterMask>
  CUTLASS_DEVICE void
  mma(Params const& mainloop_params, MainloopPipeline pipeline, PipelineState slm_pipe_read, FinalPipeline finalPipeline, FinalPipelineState& finalPipelineState, FrgTensorC& accumulator, int k_tile_count, int local_id, ClusterMask const& cluster_mask, TensorStorage& shared_tensors) {
    static_assert(cute::rank(SmemLayoutA{}) == 3, "Smem layout must be rank 3.");
    static_assert(cute::rank(SmemLayoutB{}) == 3, "Smem layout must be rank 3.");
    static_assert(cute::is_void_v<SmemCopyAtomA>,
      "XE4 GMMA mainloops cannot have a non-void copy atom for smem sourced instructions.");
    static_assert(cute::is_void_v<SmemCopyAtomB>,
      "XE4 GMMA mainloops cannot have a non-void copy atom for smem sourced instructions.");

    auto sA = make_tensor(reinterpret_cast<ElementA *>(shared_tensors.smem_A.data()), SmemLayoutA {});
    auto sB = make_tensor(reinterpret_cast<ElementB *>(shared_tensors.smem_B.data()), SmemLayoutB {});

    auto [load_a, load_b] = mainloop_params;
    auto [cluster_mask_a, cluster_mask_b] = cluster_mask;

    TiledMma tiled_mma;
    constexpr auto scaleOutOne = C<cute::xe4::AMMA::ScaleOut::One>{};
    constexpr auto scaleOutZero = C<cute::xe4::AMMA::ScaleOut::Zero>{};
    constexpr auto dstType = C<cute::xe4::AMMA::DstType::Accum>{};

    auto thread_mma = tiled_mma.get_thread_slice(0);
    auto tCsA = thread_mma.partition_fragment_A(sA);            // (MMA,MMA_M,MMA_K,PIPE)
    auto tCsB = thread_mma.partition_fragment_B(sB);            // (MMA,MMA_N,MMA_K,PIPE)
    auto accum = thread_mma.partition_fragment_C(accumulator);  // (MMA,MMA_M,MMA_N)

    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(accum));                           // M
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<2>(accum));                           // N
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));                            // K
    CUTE_STATIC_ASSERT_V(size<3>(tCsA) == size<3>(tCsB));                         // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA));           // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sB));           // PIPE

    auto cshape = ClusterShape{};
    auto wg_expect_tx = size<1>(accum) * size<2>(accum) * size<2>(tCsA);
    auto cluster_expect_tx = wg_expect_tx * (size<0>(cshape) + size<1>(cshape));

    pipeline.consumer_try_wait(slm_pipe_read);
    uint32_t read_stage = slm_pipe_read.index();
    auto abar_cons = pipeline.consumer_get_barrier(slm_pipe_read);
    cute::gemm(tiled_mma.with(scaleOutZero, dstType, abar_cons, cluster_mask_a, abar_cons, cluster_mask_b), tCsA(_,_,0,read_stage), tCsB(_,_,0,read_stage), accum);
    for (int k_block = 1; k_block < size<2>(tCsA); ++k_block) {
      cute::gemm(tiled_mma.with(scaleOutOne, dstType, abar_cons, cluster_mask_a, abar_cons, cluster_mask_b), tCsA(_,_,k_block,read_stage), tCsB(_,_,k_block,read_stage), accum);
    }
    pipeline.consumer_commit(slm_pipe_read, cluster_expect_tx);
    ++slm_pipe_read;

    for (uint32_t i = 1; i < k_tile_count-1; ++i, ++slm_pipe_read) {
      uint32_t read_stage = slm_pipe_read.index();
      pipeline.consumer_try_wait(slm_pipe_read);
      auto abar_cons = pipeline.consumer_get_barrier(slm_pipe_read);
      cute::gemm(tiled_mma.with(scaleOutOne, dstType, abar_cons, cluster_mask_a, abar_cons, cluster_mask_b), tCsA(_,_,_,read_stage), tCsB(_,_,_,read_stage), accum);
      pipeline.consumer_commit(slm_pipe_read, cluster_expect_tx);
    }

    {
      uint32_t read_stage = slm_pipe_read.index();
      pipeline.consumer_try_wait(slm_pipe_read);
      auto abar_cons = pipeline.consumer_get_barrier(slm_pipe_read);
      auto abar_cons_d = finalPipeline.store_get_barrier(finalPipelineState);
      cute::gemm(tiled_mma.with(scaleOutOne, dstType, abar_cons_d, abar_cons, cluster_mask_a, abar_cons, cluster_mask_b), tCsA(_,_,_,read_stage), tCsB(_,_,_,read_stage), accum);
      pipeline.consumer_commit(slm_pipe_read, cluster_expect_tx);
      finalPipeline.store_commit(finalPipelineState, 1);
      finalPipeline.store_try_wait(finalPipelineState);
      ++finalPipelineState;
    }
  }
};

}