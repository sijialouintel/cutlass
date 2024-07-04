#pragma once

#include "xe4_mma_traits.hpp"
#include "xe4_copy_traits.hpp"
#include "inline_pisa.hpp"
#include "xe4_mma.hpp"
#include "pipe.hpp"

namespace cutlass::gemm::collective {
using namespace cute;
using namespace cute::detail;

template <
  int Stages,
  class TileShape_,
  class ElementA_,
  class StrideA_,
  class ElementB_,
  class StrideB_,
  class ElementC_,
  class ElementAcc_,
  class StrideC_,
  class GmemTiledCopyA_,
  class SmemLayoutAtomA_,
  class GmemTiledCopyB_,
  class SmemLayoutAtomB_,
  class GmemTiledCopyC_,
  class SmemLayoutAtomC_,
  class TensorDescPtr_,
  class AbarrierPtr_,
  class MatrixDesc_
>
struct CollectiveMma {

  using TileShape = TileShape_;
  using ElementA = ElementA_;
  using StrideA = StrideA_;
  using ElementB = ElementB_;
  using StrideB = StrideB_;
  using ElementC = ElementC_;
  using ElementAcc = ElementAcc_;
  using StrideC = StrideC_;
  using GmemTiledCopyA = GmemTiledCopyA_;
  using GmemTiledCopyB = GmemTiledCopyB_;
  using GmemTiledCopyC = GmemTiledCopyC_;

  using SmemLayoutA = SmemLayoutAtomA_;
  using SmemLayoutB = SmemLayoutAtomB_;
  using SmemLayoutC = SmemLayoutAtomC_;

  using TensorDescPtr = TensorDescPtr_;
  using AbarrierPtr = AbarrierPtr_;
  using MatrixDesc = MatrixDesc_;

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
        SmemLayoutB{}, make_shape(shape<2>(TileShape{}), shape<1>(TileShape{}))));

    using TiledStoreC = decltype(make_xe4_copy<GmemTiledCopyC, AuxParamsC>(
      make_tensor(static_cast<ElementC const*>(nullptr), repeat_like(StrideC{}, int32_t(0)), StrideC{}),
      SmemLayoutC{}, make_shape(shape<1>(TileShape{}), shape<0>(TileShape{}))));

    TiledLoadA load_a;
    TiledLoadB load_b;
    TiledStoreC store_c;

    using SlmTensorA = decltype(make_tensor(static_cast<ElementA*>(nullptr), SmemLayoutA{}));
    using SlmTensorB = decltype(make_tensor(static_cast<ElementB*>(nullptr), SmemLayoutB{}));
    using SlmTensorC = decltype(make_tensor(static_cast<ElementC*>(nullptr), SmemLayoutC{}));

    SlmTensorA slm_a;
    SlmTensorB slm_b;
    SlmTensorC slm_c;
  };;

  template<class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args) {
    auto [M, N, K] = problem_shape;

    auto A = make_tensor(args.ptr_A, make_layout(make_shape(M,K), args.dA));
    auto B = make_tensor(args.ptr_B, make_layout(make_shape(K,N), args.dB));
    auto C = make_tensor(args.ptr_C, make_layout(make_shape(M,N), args.dC));

    auto load_a = make_xe4_copy<GmemTiledCopyA, AuxParamsA>(A, SmemLayoutA{}, make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})));
    auto load_b = make_xe4_copy<GmemTiledCopyB, AuxParamsB>(B, SmemLayoutB{}, make_shape(shape<2>(TileShape{}), shape<1>(TileShape{})));
    auto store_c = make_xe4_copy<GmemTiledCopyC, AuxParamsC>(C, SmemLayoutC{}, make_shape(shape<0>(TileShape{}), shape<1>(TileShape{})));

    auto [slm_a, slm_b, slm_c] = allocate_share_local_memory(args);

    return {load_a, load_b, store_c, slm_a, slm_b, slm_c};
  }

  static constexpr auto
  allocate_share_local_memory(Arguments const& args) {
    constexpr auto slm_bytes = sizeof(ElementA)*size(SmemLayoutA{}) + sizeof(ElementB)*size(SmemLayoutB{}) + sizeof(ElementC)*size(SmemLayoutC{});
    auto ptr = sycl::ext::oneapi::group_local_memory_for_overwrite<uint8_t[slm_bytes]>(args.group);

    auto slm_a = make_tensor(reinterpret_cast<ElementA*>(*ptr), SmemLayoutA{});
    auto slm_b = make_tensor(reinterpret_cast<ElementB*>(slm_a.data()+size(SmemLayoutA{})), SmemLayoutB{});
    auto slm_c = make_tensor(reinterpret_cast<ElementC*>(slm_b.data()+size(SmemLayoutB{})), SmemLayoutC{});

    return std::make_tuple(slm_a, slm_b, slm_c);
  }

  template <class ProblemShape>
  CUTLASS_DEVICE auto
  load_init(ProblemShape const& problem_shape, Params const& mainloop_params) const {
    auto [M, N, K] = problem_shape;

    auto mA_mk = mainloop_params.load_a.get_tma_tensor(make_shape(M, K));   // (m,k)
    auto mB_kn = mainloop_params.load_b.get_tma_tensor(make_shape(K, N));   // (k,n)
    auto mC_mn = mainloop_params.store_c.get_tma_tensor(make_shape(M, N));  // (m,n)

    auto gA_mk = flat_divide(mA_mk, make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})));  // (BLK_M,BLK_K,m,k)
    auto gB_kn = flat_divide(mB_kn, make_shape(shape<2>(TileShape{}), shape<1>(TileShape{})));  // (BLK_K,BLK_N,k,n)
    auto gC_mn = flat_divide(mC_mn, make_shape(shape<0>(TileShape{}), shape<1>(TileShape{})));  // (BLK_M,BLK_N,m,n)

    return cute::make_tuple(gA_mk, gB_kn, gC_mn);
  }

  template <class TensorA, class TensorB, class TensorC, class BlockCoord>
  CUTLASS_DEVICE void
  load(Params const& mainloop_params, MainloopPipeline pipeline, PipelineState slm_pipe_write,
    cute::tuple<TensorA, TensorB, TensorC> const& load_inputs, BlockCoord const& blk_coord, int k_tile_count, int local_id) {
    if(local_id == 0){
      sycl::vec<uint32_t, 2> elem_stride(1, 1);

      auto [load_a, load_b, store_c, sA, sB, sC] = mainloop_params;

      auto block_load_a = load_a.get_slice(0);
      auto block_load_b = load_b.get_slice(0);
      auto block_load_c = store_c.get_slice(0);

      auto [gA_mk, gB_kn, gC_mn] = load_inputs;
      auto [m_coord, n_coord] = blk_coord;

      auto gA = gA_mk(_, _, m_coord, _);          // (BLK_M,BLK_K,k)
      auto tAgA = block_load_a.partition_S(gA);   // (TMA,TMA_M,TMA_K,k)
      auto tAsA = block_load_a.partition_D(sA);   // (TMA,TMA_M,TMA_K,PIPE)

      auto gB = gB_kn(_, _, _, n_coord);          // (BLK_K,BLK_N,k)
      auto tBgB = block_load_b.partition_S(gB);   // (TMA,TMA_K,TMA_N,k)
      auto tBsB = block_load_b.partition_D(sB);   // (TMA,TMA_K,TMA_N,PIPE)

      auto gC = gC_mn(_, _, m_coord, n_coord);    // (BLK_M,BLK_N)
      auto tCgC = block_load_c.partition_S(gC);   // (TMA,TMA_M,TMA_N)
      auto tCsC = block_load_c.partition_D(sC);   // (TMA,TMA_M,TMA_N)

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

  CUTLASS_DEVICE void
  mma(Params const& mainloop_params, MainloopPipeline pipeline, PipelineState slm_pipe_read, int k_tile_count, int local_id) {
    if (local_id == 32) {
      auto [load_a, load_b, store_c, sA, sB, sC] = mainloop_params;

      auto tiled_mma = cute::make_tiled_mma(MMA_Op<ElementAcc, void>{});

      auto thread_mma = tiled_mma.get_thread_slice(0);
      auto tCrA = thread_mma.partition_fragment_A(sA);    // (MMA,MMA_M,MMA_K,PIPE)
      auto tCrB = thread_mma.partition_fragment_B(sB);    // (MMA,MMA_K,MMA_N,PIPE)
      auto tCrC = thread_mma.partition_fragment_C(sC);    // (MMA,MMA_N,MMA_N)

      if (k_tile_count == 1) {
        pipeline.consumer_try_wait(slm_pipe_read);
        auto abar_cons = pipeline.consumer_get_barrier(slm_pipe_read);
        cute::gemm(tiled_mma.with(abar_cons, MMA_Op<ElementAcc, void>{}), tCrA(_,_,_,0), tCrB(_,_,_,0), tCrC);
        pipeline.consumer_commit(slm_pipe_read);
        return;
      }

      if (k_tile_count >= 2) {
        PipelineState slm_pipe_read;
        pipeline.consumer_try_wait(slm_pipe_read);
        uint32_t index = slm_pipe_read.index();
        auto abar_cons = pipeline.consumer_get_barrier(index);
        cute::gemm(tiled_mma.with(abar_cons, MMA_Op<ElementAcc, void>{}), tCrA(_,_,_,index), tCrB(_,_,_,index), tCrC);
        pipeline.consumer_commit(slm_pipe_read);

        for (uint32_t i = 1; i < k_tile_count - 1; i++) {
            ++slm_pipe_read;
            uint32_t index = slm_pipe_read.index();
            auto abar_cons = pipeline.consumer_get_barrier(index);
            pipeline.consumer_try_wait(slm_pipe_read);
            cute::gemm(tiled_mma.with(abar_cons, MMA_Op<ElementAcc, ElementAcc>{}), tCrA(_,_,_,index), tCrB(_,_,_,index), tCrC);
            pipeline.consumer_commit(slm_pipe_read);
        }
        {
            ++slm_pipe_read;
            uint32_t index = slm_pipe_read.index();

            auto abar_cons = pipeline.consumer_get_barrier(index);
            auto abar_prod = pipeline.producer_get_barrier(index);

            uint32_t phase = ((k_tile_count - 1) / Stages) & 1u;
            pipeline.consumer_try_wait(index, phase);

            cute::gemm(tiled_mma.with(abar_cons, MMA_Op<ElementC, ElementAcc>{}), tCrA(_,_,_,index), tCrB(_,_,_,index), tCrC);
            pipeline.consumer_commit(slm_pipe_read);
        }
      }
    }
  }
};

}