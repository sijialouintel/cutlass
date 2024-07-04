#pragma once

#include "inline_pisa.hpp"

namespace cute {

template <cm_size_t cmSizeA_, cm_size_t cmSizeB_, cm_size_t cmSizeC_>
struct CoreMatrixSize {
  static constexpr cm_size_t cmSizeA = cmSizeA_;
  static constexpr cm_size_t cmSizeB = cmSizeB_;
  static constexpr cm_size_t cmSizeC = cmSizeC_;
};

template <class TD, class TC, class TA, class TB, class Shape_MNK_, class CoreMatSize_, class MatDesc=uint64_t, class Abarrier_=uint64_t*>
struct XE4_ASYNC_GMMA
{
  using DRegisters = MatDesc[1];
  using ARegisters = MatDesc[1];
  using BRegisters = MatDesc[1];
  using CRegisters = MatDesc[1];

  using Shape_MNK = Shape_MNK_;
  using Abarrier = Abarrier_;
  using CoreMatSize = CoreMatSize_;

  CUTE_HOST_DEVICE static void
  fma(Abarrier const& abar_cons,
      MatDesc const& mat_desc_d,
      MatDesc const& mat_desc_c,
      MatDesc const& mat_desc_a,
      MatDesc const& mat_desc_b)
  {
    async_gmma<TD, TC, TA, TB, get<0>(Shape_MNK{}), get<1>(Shape_MNK{}), get<2>(Shape_MNK{})>(mat_desc_d, mat_desc_c, mat_desc_a, mat_desc_b, abar_cons);
  }
};

template <class TD, class TA, class TB, class Shape_MNK_, class CoreMatSize_, class MatDesc, class Abarrier_>
struct XE4_ASYNC_GMMA<TD, void, TA, TB, Shape_MNK_, CoreMatSize_, MatDesc, Abarrier_>
{
  using DRegisters = MatDesc[1];
  using ARegisters = MatDesc[1];
  using BRegisters = MatDesc[1];
  using CRegisters = void;

  using Shape_MNK = Shape_MNK_;
  using Abarrier = Abarrier_;
  using CoreMatSize = CoreMatSize_;

  CUTE_HOST_DEVICE static void
  fma(Abarrier const& abar_cons,
      MatDesc const& mat_desc_d,
      MatDesc const& mat_desc_c,
      MatDesc const& mat_desc_a,
      MatDesc const& mat_desc_b)
  {
    async_gmma<TD, TA, TB, get<0>(Shape_MNK{}), get<1>(Shape_MNK{}), get<2>(Shape_MNK{})>(mat_desc_d, mat_desc_a, mat_desc_b, abar_cons);
  }
};

} // namespace cute
