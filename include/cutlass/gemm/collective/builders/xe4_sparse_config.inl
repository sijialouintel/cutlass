#pragma once

#include "cute/layout.hpp"                      // cute::Layout, cute::Shape, cute::Stride
#include "cute/pointer_sparse.hpp"              // cute::is_sparse
#include "cute/int_tuple.hpp"                   // cute::round_up
#include "cute/algorithm/tuple_algorithms.hpp"  // cute::replace

namespace cutlass {

using namespace cute;

template<
  class ElementAMma_,
  cute::xe4::GMMA::Major GmmaMajorA,
  class ElementBMma_,
  cute::xe4::GMMA::Major GmmaMajorB,
  class ElementEMma_
>
struct Xe4GemmSparseConfig {
  static_assert(cute::is_sparse<ElementAMma_>::value, "ElementAMma MUST be sparse elem");
  static_assert(cute::is_sparse<ElementEMma_>::value, "ElementEMma MUST be sparse elem");

  // A
  using ElementAMma         = ElementAMma_;
  using ElementAMmaRaw      = typename ElementAMma::raw_type;
  using ElementAMmaSparsity = Int<ElementAMma::sparsity>;

  // B
  using ElementBMma         = ElementBMma_;
  using ElementBMmaRaw      = typename ElementBMma::raw_type;
  using ElementBMmaSparsity = Int<ElementBMma::sparsity>;

  // E
  using ElementEMma         = ElementEMma_;
  using ElementEMmaRaw      = typename ElementEMma::raw_type;
  using ElementEMmaSparsity = Int<ElementEMma::sparsity>;

  using ElementASparsity = ElementAMmaSparsity;
  using ElementBSparsity = ElementBMmaSparsity;
  using ElementESparsity = ElementEMmaSparsity;

  static constexpr uint32_t MXScaleElemNum = 32;

  // The following two functions are provided for user determine the static layouts type
  CUTE_HOST_DEVICE
  static constexpr auto
  deduce_layoutA() {
    static_assert(GmmaMajorA == cute::xe4::GMMA::Major::K, "Only GMMA::Major::K is supported for A");

    using LayoutMMajor = Layout<Shape <int32_t,
                                       Shape<ElementASparsity, int32_t>,
                                       int32_t>,
                                Stride<ElementASparsity,
                                       Stride<_1, int64_t>,
                                       int64_t>>;

    using LayoutKMajor = Layout<Shape <int32_t,
                                       Shape<ElementASparsity, int32_t>,
                                       int32_t>,
                                Stride<int64_t,
                                       Stride<_1, ElementASparsity>,
                                       int64_t>>;

    if constexpr (GmmaMajorA == cute::xe4::GMMA::Major::MN) {
      return LayoutMMajor{};
    }
    else {
      return LayoutKMajor{};
    }
  }

  CUTE_HOST_DEVICE
  static constexpr auto
  deduce_layoutB() {
    using LayoutNMajor = Layout<Shape <int32_t,
                                       Shape<ElementASparsity, int32_t>,
                                       int32_t>,
                                Stride<ElementBSparsity,
                                       Stride<_1, int64_t>,
                                       int64_t>>;

    using LayoutKMajor = Layout<Shape <int32_t,
                                       Shape<ElementBSparsity, int32_t>,
                                       int32_t>,
                                Stride<int64_t,
                                       Stride<_1, ElementBSparsity>,
                                       int64_t>>;

    if constexpr (GmmaMajorB == cute::xe4::GMMA::Major::MN) {
      return LayoutNMajor{};
    }
    else {
      return LayoutKMajor{};
    }
  }

  template <typename WgK>
  CUTE_HOST_DEVICE
  static constexpr auto
  deduce_layoutMeta(WgK const& wgK) {
    return Layout<Shape<int32_t, Shape<ElementESparsity, int32_t>, int32_t>, Stride<ElementESparsity, Stride<_1, int64_t>, int64_t>>{};
  }

  template <typename WgK>
  CUTE_HOST_DEVICE
  static constexpr auto
  deduce_wgMetaKStep(WgK const&) {
    return WgK{} / Int<MXScaleElemNum>{};
  }

  template <typename WgK>
  CUTE_HOST_DEVICE
  static constexpr auto
  deduce_wgMetaK(WgK const& wgK) {
    return round_up(deduce_wgMetaKStep(wgK), Int<32>{});
  }

  template <typename TileShape>
  CUTE_HOST_DEVICE
  static constexpr auto
  deduce_MetaTileShape() {
    return replace<2>(TileShape{}, deduce_wgMetaK(get<2>(TileShape{})));
  }

  // Fill tensor A layout from dynamic problem shape
  template <class ProblemShape>
  CUTE_HOST_DEVICE
  static constexpr auto
  fill_layoutA(ProblemShape problem_shape) {
    const auto [M, N, K, L] = problem_shape;

    static_assert(GmmaMajorA == cute::xe4::GMMA::Major::K, "Only GMMA::Major::K is supported for A");

    if constexpr (GmmaMajorA == cute::xe4::GMMA::Major::MN) {
      return make_layout(
        make_shape(int32_t(M),
                   make_shape(ElementASparsity{}, int32_t(K) / ElementASparsity{}),
                   int32_t(L)),
        make_stride(ElementASparsity{},
                    make_stride(_1{}, int64_t(M) * ElementASparsity{}),
                    (L == 1) ? int64_t(0) : int64_t(M * K))
      );
    }
    else {
      return make_layout(
        make_shape(int32_t(M),
                   make_shape(ElementASparsity{}, int32_t(K / ElementASparsity{})),
                   int32_t(L)),
        make_stride(int64_t(K),
                    make_stride(_1{}, ElementASparsity{}),
                    (L == 1) ? int64_t(0) : int64_t(M * K))
      );
    }
  }

  // Fill tensor B layout from dynamic problem shape
  template <class ProblemShape>
  CUTE_HOST_DEVICE
  static constexpr auto
  fill_layoutB(ProblemShape problem_shape) {
    const auto [M, N, K, L] = problem_shape;

    if constexpr (GmmaMajorB == cute::xe4::GMMA::Major::MN) {
      return make_layout(
        make_shape(int32_t(N),
                    make_shape(ElementBSparsity{}, int32_t(K / ElementBSparsity{})),
                    int32_t(L)),
        make_stride(ElementBSparsity{},
                    make_stride(_1{}, int64_t(N) * ElementBSparsity{}),
                    (L == 1) ? int64_t(0) : int64_t(N * K))
      );
    } else {
      return make_layout(
        make_shape(int32_t(N),
                    make_shape(ElementBSparsity{}, int32_t(K / ElementBSparsity{})),
                    int32_t(L)),
        make_stride(int64_t(K),
                    make_stride(_1{}, ElementBSparsity{}),
                    (L == 1) ? int64_t(0) : int64_t(N * K))
      );
    }
  }

  // Fill tensor MetaA layout from dynamic problem shape
  template <class ProblemShape>
  CUTE_HOST_DEVICE
  static constexpr auto
  fill_layout_metaA(ProblemShape problem_shape) {
    const auto [M, N, K, L] = problem_shape;
    int32_t metaK = std::max(32u, K / MXScaleElemNum);

    return make_layout(
      make_shape(int32_t(M), make_shape(ElementESparsity{}, int32_t(K / ElementESparsity{})), int32_t(L)),
      make_stride(ElementESparsity{}, make_stride(_1{}, int64_t(M) * ElementESparsity{}), (L == 1) ? int64_t(0) : int64_t(M * K))
    );
  }

  // Fill tensor MetaB layout from dynamic problem shape
  template <class ProblemShape>
  CUTE_HOST_DEVICE
  static constexpr auto
  fill_layout_metaB(ProblemShape problem_shape) {
    const auto [M, N, K, L] = problem_shape;
    int32_t metaK = std::max(32u, K / MXScaleElemNum);

    return make_layout(
      make_shape(int32_t(N), make_shape(ElementESparsity{}, int32_t(K / ElementESparsity{})), int32_t(L)),
      make_stride(ElementESparsity{}, make_stride(_1{}, int64_t(N) * ElementESparsity{}), (L == 1) ? int64_t(0) : int64_t(N * K))
    );
  }

};

}
