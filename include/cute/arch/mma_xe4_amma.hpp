#pragma once

#include "inline_pisa.hpp"

namespace cute {

namespace xe4::GMMA {

enum class Major {
  K  = 0,
  MN = 1
};

enum class ScaleOut {
  Zero = 0,
  One  = 1
};

enum class DstType {
  Accum = 0,
  MatC = 1
};
}

template <class TD, class TC, class TA, class TB, class Shape_MNK_, xe4::GMMA::Major tnspA, xe4::GMMA::Major tnspB, class MatDesc, class Abarrier_=uint64_t*>
struct XE4_ASYNC_GMMA
{
  using DRegisters = MatDesc[1];
  using ARegisters = MatDesc[1];
  using BRegisters = MatDesc[1];
  using CRegisters = MatDesc[1];

  using Shape_MNK = Shape_MNK_;
  using Abarrier = Abarrier_;

  template<typename ConstScaleOut, typename ConstDstType, typename... Args>
  CUTE_HOST_DEVICE static void
  fma(MatDesc const& mat_desc_d,
      MatDesc const& mat_desc_c,
      MatDesc const& mat_desc_a,
      MatDesc const& mat_desc_b,
      ConstScaleOut const& scale_D,
      ConstDstType const& dst_type,
      Args&&... args)
  {
    constexpr auto Tile_M = get<0>(Shape_MNK{});
    constexpr auto Tile_N = get<1>(Shape_MNK{});
    constexpr auto Tile_K = get<2>(Shape_MNK{});

    constexpr mem_layout layout_a = (tnspA == xe4::GMMA::Major::K) ? mem_layout::row_major: mem_layout::col_major;
    constexpr mem_layout layout_b = (tnspB == xe4::GMMA::Major::MN) ? mem_layout::row_major: mem_layout::col_major;

    using TDst = std::conditional_t<ConstDstType::value == xe4::GMMA::DstType::MatC, TD, TC>;
    auto& mat_desc_dst = (ConstDstType::value == xe4::GMMA::DstType::MatC) ? mat_desc_d : mat_desc_c;
    if constexpr (ConstScaleOut::value == xe4::GMMA::ScaleOut::Zero) {
      async_gmma<TDst, TA, TB, Tile_M, Tile_N, Tile_K, layout_a, layout_b>(mat_desc_dst, mat_desc_a, mat_desc_b, static_cast<Args&&>(args)...);
    } else {
      async_gmma<TDst, TC, TA, TB, Tile_M, Tile_N, Tile_K, layout_a, layout_b>(mat_desc_dst, mat_desc_c, mat_desc_a, mat_desc_b, static_cast<Args&&>(args)...);
    }
  }
};

template <class TD, class TC, class TA, class TB, class Shape_MNK_, xe4::GMMA::Major tnspA, xe4::GMMA::Major tnspB, class MatDesc, class Abarrier_=uint64_t*>
struct XE4_ASYNC_GMMA_MULTICAST
{
  using DRegisters = MatDesc[1];
  using ARegisters = MatDesc[1];
  using BRegisters = MatDesc[1];
  using CRegisters = MatDesc[1];

  using Shape_MNK = Shape_MNK_;
  using Abarrier = Abarrier_;

  template<typename ConstScaleOut, typename ConstDstType, typename... Args>
  CUTE_HOST_DEVICE static void
  fma(MatDesc const& mat_desc_d,
      MatDesc const& mat_desc_c,
      MatDesc const& mat_desc_a,
      MatDesc const& mat_desc_b,
      ConstScaleOut const& scale_D,
      ConstDstType const& dst_type,
      Args&&... args)
  {

    constexpr auto Tile_M = get<0>(Shape_MNK{});
    constexpr auto Tile_N = get<1>(Shape_MNK{});
    constexpr auto Tile_K = get<2>(Shape_MNK{});

    constexpr mem_layout layout_a = (tnspA == xe4::GMMA::Major::K) ? mem_layout::row_major: mem_layout::col_major;
    constexpr mem_layout layout_b = (tnspB == xe4::GMMA::Major::MN) ? mem_layout::row_major: mem_layout::col_major;

    using TDst = std::conditional_t<ConstDstType::value == xe4::GMMA::DstType::MatC, TD, TC>;
    auto& mat_desc_dst = (ConstDstType::value == xe4::GMMA::DstType::MatC) ? mat_desc_d : mat_desc_c;
    if constexpr (ConstScaleOut::value == xe4::GMMA::ScaleOut::Zero) {
      async_gmma<TDst, TA, TB, Tile_M, Tile_N, Tile_K, layout_a, layout_b>(mat_desc_dst, mat_desc_a, mat_desc_b, static_cast<Args&&>(args)...);
    } else {
      async_gmma<TDst, TC, TA, TB, Tile_M, Tile_N, Tile_K, layout_a, layout_b>(mat_desc_dst, mat_desc_c, mat_desc_a, mat_desc_b, static_cast<Args&&>(args)...);
    }
  }
};

} // namespace cute