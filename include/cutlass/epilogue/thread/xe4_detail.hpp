#pragma once

#include "cute/tensor.hpp"
#include "cute/util/print.hpp"
#include "cute/arch/copy_xe4_dma.hpp"
#include "cute/atom/copy_traits_xe4_dma.hpp"

namespace cutlass {
namespace epilogue {
namespace thread {
namespace detail {

using namespace cute;

struct CoreMatrix {
  static constexpr int kRowsPerCmTile = 2;
  static constexpr int kEsubBanksPerBank = 4;
  static constexpr int kSlmBanks = 4;
  static constexpr int kBytesPerCmRow = 32;

  static constexpr auto kCmLayoutRaw = make_ordered_layout(
    Shape<Shape<Int<kRowsPerCmTile>,Int<kEsubBanksPerBank>,Int<kSlmBanks>>,Int<kBytesPerCmRow>>{},
    Step<Step<_1,_3,_2>,_0>{}
  );

  template<typename ValType, typename TileShape>
  CUTLASS_HOST_DEVICE static constexpr auto retile(TileShape const& tile_shape) {
    auto cm_layout = recast_layout<uint8_t, ValType>(kCmLayoutRaw);
    auto retiled_layout = tile_to_shape(cm_layout, tile_shape, Step<_1,_0>{});
    auto swizzled_layout = composition(make_swizzle<ValType>(), retiled_layout);
    return swizzled_layout;
  }

  template<typename ValType>
  CUTLASS_HOST_DEVICE static constexpr auto make_swizzle() {
    constexpr int kSwizzleB = 1;
    constexpr int kSwizzleM = countr_zero(kBytesPerCmRow / sizeof(ValType));
    constexpr int kSwizzleS = countr_zero(size(kCmLayoutRaw) / sizeof(ValType)) - kSwizzleM;
    return Swizzle<1, kSwizzleM, kSwizzleS>{};
  }
};

enum class EpilogueAccessPattern {
  Pattern1, // TODO: Each subgroup processes one core matrix
  Pattern2, // Each subgroup processes one or more rows of data across multiple core matrices
  Pattern3, // TODO
};

template <template <uint32_t> class SlmVOp, class ValType, int SubgroupNum, int SubgroupSize, class TileShape>
CUTLASS_HOST_DEVICE constexpr auto make_pattern2_tiled_copy(TileShape const& tile_shape) {
  static_assert(is_static<TileShape>::value, "Tile shape must be static");

  constexpr int tile_M = CUTE_STATIC_V(get<0>(tile_shape));
  constexpr int tile_N = CUTE_STATIC_V(get<1>(tile_shape));

  constexpr int kMaxLanesPerRow = 16;
  constexpr int kMaxBytesPerLoad = 32;
  constexpr int maxValuesPerLoad = kMaxBytesPerLoad / sizeof(ValType);
  static_assert(tile_N % maxValuesPerLoad == 0, "tile_N must be divisible by maxValuesPerLoad!");
  /**
   * @brief Use as many as lanes to process one row then calculate the minimum rows a subgroups would
   * process (`minRowsPerSubgroup`).
   */
  constexpr int maxLanesPerRow = cute::min(tile_N / maxValuesPerLoad, kMaxLanesPerRow);
  static_assert(cute::popcount(maxLanesPerRow) == 1, "maxLanesPerRow must be a power of 2!");
  constexpr int minRowsPerSubgroupPerIter = SubgroupSize / maxLanesPerRow;

  static_assert(tile_M % SubgroupNum == 0, "tile_M must be divisible by SubgroupNum!");
  constexpr int totalRowsPerSubgroup = tile_M / SubgroupNum;

  /**
   * @brief Lanes are supposed to load SLM with `kMaxBytesPerLoad`, but if this cause some subgroups
   * in idle, lanes could load less than `kMaxBytesPerLoad` bytes.
   */
  constexpr int numRowsPerSubgroup = cute::min(totalRowsPerSubgroup, minRowsPerSubgroupPerIter);
  static_assert(tile_M % numRowsPerSubgroup == 0, "tile_M must be divisible by numRowsPerSubgroup!");
  static_assert(cute::popcount(numRowsPerSubgroup) == 1, "numRowsPerSubgroup must be a power of 2!");

  constexpr int numLanesPerRow = SubgroupSize / numRowsPerSubgroup;
  static_assert(tile_N % numLanesPerRow == 0, "tile_N must be divisible by numLanesPerRow!");
  constexpr int numValuesPerLane = cute::min(tile_N / numLanesPerRow, maxValuesPerLoad);

  auto thr_layout = make_ordered_layout(
    Shape<Shape<Int<numRowsPerSubgroup>, Int<SubgroupNum>>, Int<numLanesPerRow>>{},
    Step<Step<_0,_2>,_1>{}
  );

  auto val_layout = make_layout(Shape<_1,Int<numValuesPerLane>>{}, GenRowMajor{});

  using Copy_Traits = Copy_Traits<SlmVOp<numValuesPerLane>, Int<8 * numValuesPerLane * sizeof(ValType)>>;
  using Atom = Copy_Atom<Copy_Traits, ValType>;
  auto tiled_copy = make_tiled_copy(Atom{}, thr_layout, val_layout);

  return tiled_copy;
}

/**
 * @brief We can't use `make_tensor_like()` here due to the alignment issue of allocating registers.
 */
template <typename Tensor>
CUTLASS_HOST_DEVICE constexpr auto make_register_tensor(Tensor const& tensor) {
  using Layout = decltype(make_layout_like(tensor.layout()));
  using Engine = Xe4Engine<typename Tensor::value_type, cosize_v<Layout>>;
  return cute::Tensor<Engine, Layout>();
}

template <
  int FragmentSize,
  int SubgroupNum,
  int SubgroupSize,
  typename CstCallbacks,
  typename STensor,
  typename DTensor
>
CUTLASS_HOST_DEVICE
void pattern2(CstCallbacks& cst_callbacks, STensor const& src_tensor, DTensor& dst_tensor, uint32_t worker_id) {
  using SType = typename STensor::value_type;
  using DType = typename DTensor::value_type;

  auto tile_shape = product_each(shape(dst_tensor));

  auto tiled_s2r = make_pattern2_tiled_copy<cute::xe4::SLM_VLOAD, SType, SubgroupNum, SubgroupSize>(tile_shape);
  auto thread_s2r = tiled_s2r.get_thread_slice(worker_id);
  Tensor tSR_src = thread_s2r.partition_S(src_tensor);

  auto tiled_r2s = make_pattern2_tiled_copy<cute::xe4::SLM_VSTORE, SType, SubgroupNum, SubgroupSize>(tile_shape);
  auto thread_r2s = tiled_r2s.get_thread_slice(worker_id);
  Tensor tRS_dst = thread_r2s.partition_D(dst_tensor);

  Tensor src_v = group_modes<1,-1>(tSR_src);
  Tensor dst_v = group_modes<1,-1>(tRS_dst);

  cst_callbacks.begin();

  int epi_m = 0, epi_n = 0;

  CUTE_UNROLL
  for (int i = 0; i < size<1>(src_v); ++i) {
    auto src_r = make_register_tensor(src_v(_, _0{}));
    auto dst_r = make_register_tensor(dst_v(_, _0{}));

    copy(tiled_s2r, src_v(_, i), src_r);

    auto trSrc_frg = recast<Array<SType, FragmentSize>>(src_r);
    auto trDst_frg = recast<Array<DType, FragmentSize>>(dst_r);

    CUTE_UNROLL
    for (int epi_v = 0; epi_v < size(trSrc_frg); ++epi_v) {
      trDst_frg(epi_v) = cst_callbacks.visit(trSrc_frg(epi_v), epi_v, epi_m, epi_n);
    }

    copy(tiled_r2s, dst_r, dst_v(_, i));
  }

  cst_callbacks.end();
}

} // namespace detail
} // namespace thread
} // namespace epilogue
} // namespace cutlass
