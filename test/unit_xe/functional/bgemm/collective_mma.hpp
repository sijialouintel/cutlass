#pragma once

#include "xe4_mma_traits.hpp"
#include "xe4_copy_traits.hpp"
#include "xe4_copy_async.hpp"
#include "inline_pisa.hpp"
#include "xe4_mma.hpp"
#include "pipe.hpp"

#include "cutlass/util/packed_stride.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"

namespace cutlass::gemm::collective {

using namespace cute;
using namespace cute::detail;

template <
  class DispatchPolicy,
  class TileShape,
  class ElementA,
  class StrideA,
  class ElementB,
  class StrideB,
  class TiledMma,
  class GmemTiledCopyA,
  class SmemLayoutAtomA,
  class SmemCopyAtomA,
  class TransformA,
  class GmemTiledCopyB,
  class SmemLayoutAtomB,
  class SmemCopyAtomB,
  class TransformB
>
struct CollectiveMma;

template<
  int Stages_,
  class ClusterShape_ = Shape<_1,_1,_1>,
  class KernelSchedule = KernelTmaWarpSpecialized
>
struct MainloopXe4DmaGmma {
  constexpr static int Stages = Stages_;
  using ClusterShape = ClusterShape_;
  using ArchTag = arch::Sm90;
  using Schedule = KernelSchedule;
  static_assert(
    cute::is_same_v<Schedule, KernelTmaWarpSpecialized> ||
    cute::is_same_v<Schedule, KernelTmaWarpSpecializedPingpong> ||
    cute::is_same_v<Schedule, KernelTmaWarpSpecializedCooperative>,
    "KernelSchedule must be one of the warp specialized policies");
};

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
    MainloopXe4DmaGmma<Stages, ClusterShape, KernelSchedule>,
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

  using TileShape = TileShape_;
  using ElementA = ElementA_;
  using StrideA = StrideA_;
  using ElementB = ElementB_;
  using StrideB = StrideB_;
  using TiledMma = TiledMma_;
  using GmemTiledCopyA = GmemTiledCopyA_;
  using GmemTiledCopyB = GmemTiledCopyB_;
  using SmemLayoutA = SmemLayoutAtomA_;
  using SmemLayoutB = SmemLayoutAtomB_;

  using TensorDescPtr = uint64_t*;
  using AbarrierPtr = uint64_t*;
  using MatrixDesc = uint64_t;

  using ElementC = typename TiledMma::ValTypeD;
  using ElementAccumulator = typename TiledMma::ValTypeC;
  using StrideC = cutlass::detail::TagToStrideC_t<cutlass::layout::RowMajor>;
  using SmemLayoutC = decltype(make_layout(make_shape(shape<0>(TileShape{}), shape<1>(TileShape{})), GenRowMajor{}));
  using GmemTiledCopyC = cute::xe4::ASYNC_TENSOR_STORE;

  using AuxParamsA = AuxParams<cm_size_t::cm_32x32B, cm_layout_t::vertical_split, false, TensorDescPtr, 0>;
  using AuxParamsB = AuxParams<cm_size_t::cm_16x32B, cm_layout_t::linear, false, TensorDescPtr, 1>;
  using AuxParamsC = AuxParams<cm_size_t::cm_32x32B, cm_layout_t::vertical_split, false, TensorDescPtr, 2>;

  using MainloopPipeline = cutlass::xe4::PipelineTmaAsync<Stages, AbarrierPtr>;
  using PipelineState = cutlass::xe4::PipelineState<Stages>;

  // Host side kernel arguments
  struct Arguments {
    ElementA const* ptr_A;
    StrideA dA;
    ElementB const* ptr_B;
    StrideB dB;
    ElementC const* ptr_C;
    StrideC dC;
    sycl::group<3> group;
  };

  // Device side kernel params
  struct Params {
    using TiledLoadA = decltype(make_xe4_copy<GmemTiledCopyA, AuxParamsA>(
        make_tensor(static_cast<ElementA const*>(nullptr), repeat_like(StrideA{}, int32_t(0)), StrideA{}),
        SmemLayoutA{}, make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}))));

    using TiledLoadB = decltype(make_xe4_copy<GmemTiledCopyB, AuxParamsB>(
        make_tensor(static_cast<ElementB const*>(nullptr), repeat_like(StrideB{}, int32_t(0)), StrideB{}),
        SmemLayoutB{}, make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}))));

    using TiledStoreC = decltype(make_xe4_copy<GmemTiledCopyC, AuxParamsC>(
      make_tensor(static_cast<ElementC const*>(nullptr), repeat_like(StrideC{}, int32_t(0)), StrideC{}),
      SmemLayoutC{}, make_shape(shape<0>(TileShape{}), shape<1>(TileShape{}))));

    TiledLoadA load_a;
    TiledLoadB load_b;
    TiledStoreC store_c;

    using SlmTensorA = decltype(make_tensor(static_cast<ElementA*>(nullptr), SmemLayoutA{}));
    using SlmTensorB = decltype(make_tensor(static_cast<ElementB*>(nullptr), SmemLayoutB{}));
    using SlmTensorAcc = decltype(make_tensor(static_cast<ElementAccumulator*>(nullptr), SmemLayoutC{}));
    using SlmTensorC = decltype(make_tensor(static_cast<ElementC*>(nullptr), SmemLayoutC{}));

    SlmTensorA slm_a;
    SlmTensorB slm_b;
    SlmTensorAcc slm_acc;
    SlmTensorC slm_c;
  };

  template<class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args) {
    auto [M, N, K, L] = problem_shape;

    auto A = make_tensor(args.ptr_A, make_layout(make_shape(M,K,L), args.dA));
    auto B = make_tensor(args.ptr_B, make_layout(make_shape(N,K,L), args.dB));
    auto C = make_tensor(args.ptr_C, make_layout(make_shape(M,N,L), args.dC));

    auto load_a = make_xe4_copy<GmemTiledCopyA, AuxParamsA>(A, SmemLayoutA{}, make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})));
    auto load_b = make_xe4_copy<GmemTiledCopyB, AuxParamsB>(B, SmemLayoutB{}, make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})));
    auto store_c = make_xe4_copy<GmemTiledCopyC, AuxParamsC>(C, SmemLayoutC{}, make_shape(shape<0>(TileShape{}), shape<1>(TileShape{})));

    auto [slm_a, slm_b, slm_acc, slm_c] = allocate_share_local_memory(args);

    return {load_a, load_b, store_c, slm_a, slm_b, slm_acc, slm_c};
  }

  static constexpr auto
  allocate_share_local_memory(Arguments const& args) {
    constexpr auto slm_bytes = sizeof(ElementA)*size(SmemLayoutA{}) + sizeof(ElementB)*size(SmemLayoutB{}) + std::max(sizeof(ElementC), sizeof(ElementAccumulator))*size(SmemLayoutC{});
    auto ptr = sycl::ext::oneapi::group_local_memory_for_overwrite<uint8_t[slm_bytes]>(args.group);

    auto slm_a = make_tensor(reinterpret_cast<ElementA*>(*ptr), SmemLayoutA{});
    auto slm_b = make_tensor(reinterpret_cast<ElementB*>(slm_a.data()+size(SmemLayoutA{})), SmemLayoutB{});
    auto slm_acc = make_tensor(reinterpret_cast<ElementAccumulator*>(slm_b.data()+size(SmemLayoutB{})), SmemLayoutC{});
    auto slm_c = make_tensor(reinterpret_cast<ElementC*>(slm_b.data()+size(SmemLayoutB{})), SmemLayoutC{});

    return std::make_tuple(slm_a, slm_b, slm_acc, slm_c);
  }

  template <class ProblemShape>
  CUTLASS_DEVICE auto
  load_init(ProblemShape const& problem_shape, Params const& mainloop_params) const {
    auto [M, N, K, L] = problem_shape;

    auto mA_mkl = mainloop_params.load_a.get_tma_tensor(make_shape(M, K, L));   // (m,k,l)
    auto mB_knl = mainloop_params.load_b.get_tma_tensor(make_shape(N, K, L));   // (n,k,l)
    auto mC_mnl = mainloop_params.store_c.get_tma_tensor(make_shape(M, N, L));  // (m,n,l)

    auto gA_mkl = flat_divide(mA_mkl, make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})));  // (BLK_M,BLK_K,m,k,l)
    auto gB_knl = flat_divide(mB_knl, make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})));  // (BLK_N,BLK_K,n,k,l)
    auto gC_mnl = flat_divide(mC_mnl, make_shape(shape<0>(TileShape{}), shape<1>(TileShape{})));  // (BLK_M,BLK_N,m,n,l)

    return cute::make_tuple(gA_mkl, gB_knl, gC_mnl);
  }

  template <class TensorA, class TensorB, class TensorC, class BlockCoord>
  CUTLASS_DEVICE void
  load(Params const& mainloop_params, MainloopPipeline pipeline, PipelineState slm_pipe_write,
    cute::tuple<TensorA, TensorB, TensorC> const& load_inputs, BlockCoord const& blk_coord, int k_tile_count, int local_id) {
    if(local_id == 0){
      auto [load_a, load_b, store_c, sA, sB, sAcc, sC] = mainloop_params;

      auto block_load_a = load_a.get_slice(0);
      auto block_load_b = load_b.get_slice(0);
      auto block_load_c = store_c.get_slice(0);

      auto [gA_mkl, gB_knl, gC_mnl] = load_inputs;
      auto [m_coord, n_coord, l_coord] = blk_coord;

      auto gA = gA_mkl(_, _, m_coord, _, l_coord);        // (BLK_M,BLK_K,k)
      auto tAgA = block_load_a.partition_S(gA);           // (TMA,TMA_M,TMA_K,k)
      auto tAsA = block_load_a.partition_D(sA);           // (TMA,TMA_M,TMA_K,PIPE)

      auto gB = gB_knl(_, _, n_coord, _, l_coord);        // (BLK_N,BLK_K,k)
      auto tBgB = block_load_b.partition_S(gB);           // (TMA,TMA_N,TMA_K,k)
      auto tBsB = block_load_b.partition_D(sB);           // (TMA,TMA_N,TMA_K,PIPE)

      auto gC = gC_mnl(_, _, m_coord, n_coord, l_coord);  // (BLK_M,BLK_N)
      auto tCgC = block_load_c.partition_S(gC);           // (TMA,TMA_M,TMA_N)
      auto tCsC = block_load_c.partition_D(sC);           // (TMA,TMA_M,TMA_N)

      constexpr uint32_t slm_bytes_load = (sizeof(ElementA)*size(SmemLayoutA{}) + sizeof(ElementB)*size(SmemLayoutB{})) / Stages;
      constexpr uint32_t slm_bytes_store =  sizeof(ElementC)*size(SmemLayoutC{});

      for (int i = 0; i < k_tile_count; ++i, ++slm_pipe_write) {
        pipeline.producer_try_wait(slm_pipe_write);

        uint32_t index = slm_pipe_write.index();
        auto abar_prod = pipeline.producer_get_barrier(index);

        copy(load_a.with(abar_prod), tAgA(_,_,_,i), tAsA(_,_,_,index));
        copy(load_b.with(abar_prod), tBgB(_,_,_,i), tBsB(_,_,_,index));

        pipeline.producer_commit(index, slm_bytes_load);
      }

      if (k_tile_count > 0) {   // store out
        auto slm_pipe_state = PipelineState::make_pipeline_state({}, k_tile_count - 1);

        pipeline.producer_try_wait(slm_pipe_state);

        ++slm_pipe_state;
        auto abar_prod = pipeline.producer_get_barrier(slm_pipe_state);

        copy(store_c.with(abar_prod), tCsC, tCgC);

        pipeline.producer_commit(slm_pipe_state, slm_bytes_store);
        pipeline.consumer_try_wait(slm_pipe_state);
      }
    }
  }

  template<class T1, class T2>
  using MMA_Op = XE4_ASYNC_GMMA<T1, T2, ElementA, ElementB, TileShape,
    CoreMatrixSize<cm_size_t::cm_32x32B, cm_size_t::cm_16x32B, cm_size_t::cm_32x32B>, MatrixDesc, AbarrierPtr>;

  template <class FrgTensorC>
  CUTLASS_DEVICE void
  mma(Params const& mainloop_params, MainloopPipeline pipeline, PipelineState slm_pipe_read, FrgTensorC& accum, int k_tile_count, int local_id) {
    if (local_id == 32) {
      auto [load_a, load_b, store_c, sA, sB, sAcc, sC] = mainloop_params;

      TiledMma tiled_mma;
      auto thread_mma = tiled_mma.get_thread_slice(0);

      auto tCrA = thread_mma.partition_fragment_A(sA);    // (MMA,MMA_M,MMA_K,PIPE)
      auto tCrB = thread_mma.partition_fragment_B(sB);    // (MMA,MMA_N,MMA_K,PIPE)
      auto tCsC = thread_mma.partition_fragment_C(sC);    // (MMA,MMA_M,MMA_N)

      if (k_tile_count == 1) {
        pipeline.consumer_try_wait(slm_pipe_read);
        auto abar_cons = pipeline.consumer_get_barrier(slm_pipe_read);
        cute::gemm(tiled_mma.with(abar_cons, MMA_Op<ElementC, void>{}), tCrA(_,_,_,0), tCrB(_,_,_,0), tCsC);
        pipeline.consumer_commit(slm_pipe_read);
        return;
      }

      if (k_tile_count >= 2) {
        PipelineState slm_pipe_read;
        pipeline.consumer_try_wait(slm_pipe_read);
        uint32_t index = slm_pipe_read.index();
        auto abar_cons = pipeline.consumer_get_barrier(index);
        cute::gemm(tiled_mma.with(abar_cons, MMA_Op<ElementAccumulator, void>{}), tCrA(_,_,_,index), tCrB(_,_,_,index), accum);
        pipeline.consumer_commit(slm_pipe_read);

        for (uint32_t i = 1; i < k_tile_count - 1; i++) {
            ++slm_pipe_read;
            uint32_t index = slm_pipe_read.index();
            auto abar_cons = pipeline.consumer_get_barrier(index);
            pipeline.consumer_try_wait(slm_pipe_read);
            cute::gemm(tiled_mma.with(abar_cons, MMA_Op<ElementAccumulator, ElementAccumulator>{}), tCrA(_,_,_,index), tCrB(_,_,_,index), accum);
            pipeline.consumer_commit(slm_pipe_read);
        }
        {
            ++slm_pipe_read;
            uint32_t index = slm_pipe_read.index();

            auto abar_cons = pipeline.consumer_get_barrier(index);
            auto abar_prod = pipeline.producer_get_barrier(index);

            uint32_t phase = ((k_tile_count - 1) / Stages) & 1u;
            pipeline.consumer_try_wait(index, phase);

            cute::gemm(tiled_mma.with(abar_cons, MMA_Op<ElementC, ElementAccumulator>{}), tCsC, tCrA(_,_,_,index), tCrB(_,_,_,index), accum);
            pipeline.consumer_commit(slm_pipe_read);
        }
      }
    }
  }
};

}