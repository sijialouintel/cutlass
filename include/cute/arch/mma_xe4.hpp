#pragma once

#include "cute/config.hpp"

#include "inline_pisa.hpp"
#include "mma_xe4_amma.hpp"

namespace cute::xe4::GMMA {

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

template<typename T>
constexpr uint32_t getMinMmaK() {
  if constexpr (std::is_same_v<T, bf16> || std::is_same_v<T, fp16>) {
    return 16;
  }
  if constexpr (std::is_same_v<T, bf8>) {
    return 32;
  }
  if constexpr (std::is_same_v<T, fp4_e3m0x2>) {
    return 64;
  }

  CUTE_GCC_UNREACHABLE;
}

template<typename T>
constexpr uint32_t getMaxMmaK() {
  if constexpr (std::is_same_v<T, bf16> || std::is_same_v<T, fp16>) {
    return 128;
  }
  if constexpr (std::is_same_v<T, bf8>) {
    return 256;
  }
  if constexpr (std::is_same_v<T, fp4_e3m0x2>) {
    return 512;
  }

  CUTE_GCC_UNREACHABLE;
}

template <
  OpType opType,
  class ElementA,
  class ElementB,
  class ElementC,
  class ElementD,
  class TileShape_MNK,
  GMMA::Major tnspA,
  GMMA::Major tnspB,
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
  constexpr uint32_t MMA_K_MIN = cute::max(getMinMmaK<ElementA>(), getMinMmaK<ElementB>());
  constexpr uint32_t MMA_K_MAX = cute::min(getMaxMmaK<ElementA>(), getMaxMmaK<ElementB>());

  constexpr uint32_t Tile_M = size<0>(TileShape_MNK{});
  constexpr uint32_t Tile_N = size<1>(TileShape_MNK{});
  constexpr uint32_t Tile_K = size<2>(TileShape_MNK{});

  constexpr uint32_t MMA_M = gcd<Tile_M, MMA_M_MAX>::value;
  constexpr uint32_t MMA_N = gcd<Tile_N, MMA_N_MAX>::value;
  constexpr uint32_t MMA_K = gcd<Tile_K, MMA_K_MAX>::value;

  static_assert(MMA_M % 32 == 0, "MMA_M must be a multiple of 32.");
  static_assert(MMA_N % 32 == 0, "MMA_N must be a multiple of 32.");
  static_assert((MMA_K % 32 == 0) || (MMA_K == 16), "Tile_K must be a multiple of 32.");

  using MMA_Shape = Shape<Int<MMA_M>,Int<MMA_N>,Int<MMA_K>>;

  if constexpr (opType == OpType::Cluster) {
    return XE4_ASYNC_GMMA_MULTICAST<ElementD, ElementC, ElementA, ElementB, MMA_Shape, tnspA, tnspB, MatrixDesc, Abarrier>();
  } else {
    return XE4_ASYNC_GMMA<ElementD, ElementC, ElementA, ElementB, MMA_Shape, tnspA, tnspB, MatrixDesc, Abarrier>();
  }

  CUTE_GCC_UNREACHABLE;
}

} // namespace cute::xe4::GMMA