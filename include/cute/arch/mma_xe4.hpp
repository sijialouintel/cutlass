#pragma once

#include "cute/config.hpp"

#include "inline_pisa.hpp"
#include "mma_xe4_amma.hpp"

namespace cute::xe4::AMMA {

template <uint32_t a, uint32_t b>
struct gcd {
  static constexpr uint32_t value = gcd<b, a % b>::value;
};

template <uint32_t a>
struct gcd<a, 0> {
  static constexpr uint32_t value = a;
};

enum class OpType {
  Cluster,
  NoneCluster
};

template <
  OpType opType,
  class ElementA,
  class ElementB,
  class ElementC,
  class TileShape_MNK,
  bool isRowMajorA,
  bool isRowMajorB,
  auto... Args
>
CUTE_HOST_DEVICE constexpr
auto
ss_op_selector()
{
  using MatrixDesc = uint32_t;
  using Abarrier = uint64_t*;

  static_assert(is_static<TileShape_MNK>::value, "TileShape_MNK must be static.");
  static_assert(rank(TileShape_MNK{}) == 3, "TileShape_MNK must be rank 3.");

  constexpr uint32_t MMA_M_MIN = 32;
  constexpr uint32_t MMA_M_MAX = 256;
  constexpr uint32_t MMA_N_MIN = 32;
  constexpr uint32_t MMA_N_MAX = 512;

  constexpr uint32_t Tile_M = size<0>(TileShape_MNK{});
  constexpr uint32_t Tile_N = size<1>(TileShape_MNK{});
  constexpr uint32_t Tile_K = size<2>(TileShape_MNK{});

  if constexpr (is_same_v<ElementA, bf16> || is_same_v<ElementA, fp16>) {
    constexpr uint32_t MMA_K_MIN = 16;
    constexpr uint32_t MMA_K_MAX = 128;

    constexpr uint32_t MMA_M = gcd<Tile_M, MMA_M_MAX>::value;
    constexpr uint32_t MMA_N = gcd<Tile_N, MMA_N_MAX>::value;
    constexpr uint32_t MMA_K = gcd<Tile_K, MMA_K_MAX>::value;

    static_assert(MMA_M % 32 == 0, "MMA_M must be a multiple of 32.");
    static_assert(MMA_N % 32 == 0, "MMA_N must be a multiple of 32.");
    static_assert((MMA_K % 32 == 0) || (MMA_K == 16), "Tile_K must be a multiple of 32.");

    using MMA_Shape = Shape<Int<MMA_M>,Int<MMA_N>,Int<MMA_K>>;

    if constexpr (opType == OpType::Cluster) {
      return XE4_ASYNC_GMMA_MULTICAST<ElementC, ElementC, ElementB, ElementA, MMA_Shape, isRowMajorA, isRowMajorB, MatrixDesc, Abarrier>();
    } else {
      return XE4_ASYNC_GMMA<ElementC, ElementC, ElementB, ElementA, MMA_Shape, isRowMajorA, isRowMajorB, MatrixDesc, Abarrier>();
    }
  }

  CUTE_GCC_UNREACHABLE;
}

} // namespace cute::xe4::AMMA