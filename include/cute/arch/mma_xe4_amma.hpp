#pragma once

#include "inline_pisa.hpp"

namespace cute {

namespace xe4::AMMA {
enum class ScaleOut {
  Zero = 0,
  One  = 1
};
}

template <class TD, class TC, class TA, class TB, class Shape_MNK_, bool IsRowMajorA, bool IsRowMajorB, class MatDesc, class Abarrier_=uint64_t*>
struct XE4_ASYNC_GMMA
{
  using DRegisters = MatDesc[1];
  using ARegisters = MatDesc[1];
  using BRegisters = MatDesc[1];
  using CRegisters = MatDesc[1];

  using Shape_MNK = Shape_MNK_;
  using Abarrier = Abarrier_;

  template<typename ConstScaleOut>
  CUTE_HOST_DEVICE static void
  fma(ConstScaleOut const& scale_D,
      Abarrier const& abar_cons,
      MatDesc const& mat_desc_d,
      MatDesc const& mat_desc_c,
      MatDesc const& mat_desc_a,
      MatDesc const& mat_desc_b)
  {
    constexpr auto Tile_M = get<0>(Shape_MNK{});
    constexpr auto Tile_N = get<1>(Shape_MNK{});
    constexpr auto Tile_K = get<2>(Shape_MNK{});

    constexpr mem_layout layout_a = IsRowMajorA ? mem_layout::row_major: mem_layout::col_major;
    constexpr mem_layout layout_b = IsRowMajorB ? mem_layout::row_major: mem_layout::col_major;

    if constexpr (ConstScaleOut::value == xe4::AMMA::ScaleOut::One) {
      async_gmma<TD, TC, TA, TB, Tile_M, Tile_N, Tile_K, layout_a, layout_b>(mat_desc_d, mat_desc_c, mat_desc_a, mat_desc_b, abar_cons);
    } else {
      async_gmma<TD, TA, TB, Tile_M, Tile_N, Tile_K, layout_a, layout_b>(mat_desc_d, mat_desc_a, mat_desc_b, abar_cons);
    }
  }
};

template <class TD, class TC, class TA, class TB, class Shape_MNK_, bool IsRowMajorA, bool IsRowMajorB, class MatDesc, class Abarrier_=uint64_t*>
struct XE4_ASYNC_GMMA_MULTICAST
{
  using DRegisters = MatDesc[1];
  using ARegisters = MatDesc[1];
  using BRegisters = MatDesc[1];
  using CRegisters = MatDesc[1];

  using Shape_MNK = Shape_MNK_;
  using Abarrier = Abarrier_;

  template<typename ConstScaleOut>
  CUTE_HOST_DEVICE static void
  fma(ConstScaleOut const& scale_D,
      Abarrier const& abar_cons,
      uint32_t const& cluster_mask_a,
      uint32_t const& cluster_mask_b,
      MatDesc const& mat_desc_d,
      MatDesc const& mat_desc_c,
      MatDesc const& mat_desc_a,
      MatDesc const& mat_desc_b)
  {
    constexpr auto Tile_M = get<0>(Shape_MNK{});
    constexpr auto Tile_N = get<1>(Shape_MNK{});
    constexpr auto Tile_K = get<2>(Shape_MNK{});

    constexpr mem_layout layout_a = IsRowMajorA ? mem_layout::row_major: mem_layout::col_major;
    constexpr mem_layout layout_b = IsRowMajorB ? mem_layout::row_major: mem_layout::col_major;

    if constexpr (ConstScaleOut::value == xe4::AMMA::ScaleOut::One) {
      async_gmma<TD, TC, TA, TB, Tile_M, Tile_N, Tile_K, layout_a, layout_b>(
        mat_desc_d, mat_desc_c, mat_desc_a, mat_desc_b, abar_cons, cluster_mask_a, abar_cons, cluster_mask_b);
    } else {
      async_gmma<TD, TA, TB, Tile_M, Tile_N, Tile_K, layout_a, layout_b>(
        mat_desc_d, mat_desc_a, mat_desc_b, abar_cons, cluster_mask_a, abar_cons, cluster_mask_b);
    }
  }
};

} // namespace cute
