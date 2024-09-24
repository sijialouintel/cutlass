#pragma once

#include "cute/arch/mma_xe4_amma.hpp"
#include "util.hpp"

namespace cute::xe4 {

template <int LeadingRank, class MatDesc, class SEngine, class SLayout>
CUTE_HOST_DEVICE constexpr
MatDesc make_matrix_desc(Tensor<SEngine, SLayout> const& sTensor) {
  constexpr uint32_t leading_stride = get<LeadingRank, 1>(SLayout{}.stride());
  constexpr uint32_t cm_stride = (sizeof(typename SEngine::value_type) * leading_stride) >> 10;

  MatDesc mat_desc = reinterpret_cast<uint64_t>(slm_space_cast(sTensor.data())) >> 9;
  mat_desc |= (cm_stride << 16);

  return mat_desc;
}

template <class MatDesc>
struct MatDescIterator
{
  using reference    = MatDesc;
  using element_type = MatDesc;
  using value_type   = MatDesc;

  MatDesc desc_;

  MatDescIterator(MatDesc desc): desc_(desc) {}

  // Dereference returns the MatDesc
  CUTE_HOST_DEVICE constexpr
  reference operator*() const { return desc_; }

  // Advance and return a new MatDesc
  template <class Index>
  CUTE_HOST_DEVICE constexpr
  reference operator[](Index const& i) const { return *(*this + i); }

  // Return an advanced iterator
  template <class Index>
  CUTE_HOST_DEVICE constexpr
  MatDescIterator operator+(Index const& offset) const
  {
    return { MatDesc{desc_ + (MatDesc(offset) >> 9)} };
  }

  CUTE_HOST_DEVICE friend void
  print(MatDescIterator) { printf("xe4::MatDescIterator"); }
};

template <class T, class MatDesc>
CUTE_HOST_DEVICE constexpr
MatDesc
raw_pointer_cast(MatDescIterator<MatDesc> const& ptr) {
  return ptr.desc_;
}

template <int LeadingRank, class MatDesc>
struct slm_desc : MatDescIterator<MatDesc> { };

template <int M, int K>
using ABLayout = Layout<Shape<_1,Shape<Int<M>,Int<K>>>, Stride<_0,Stride<_1,Int<M>>>>;

} // end namespace cute::xe4

namespace cute {

template <int LeadingRank, class MatDesc>
struct MakeTensor<xe4::slm_desc<LeadingRank, MatDesc>>
{
  template <class TEngine, class TLayout>
  CUTE_HOST_DEVICE constexpr auto
  operator()(Tensor<TEngine,TLayout> const& smem_tensor)
  {
    auto mat_desc = xe4::make_matrix_desc<LeadingRank, MatDesc>(tensor<0>(smem_tensor));
    auto new_layout = replace<0>(recast<uint8_t const>(smem_tensor).layout(), Layout<_1,_0>{});
    return make_tensor(xe4::MatDescIterator{mat_desc}, new_layout);
  }
};

struct XE4_ASYNC_GMMA_OP {};

template <class TD, class TC, class TA, class TB, class Shape_MNK_, bool IsRowMajorA, bool IsRowMajorB, class MatDesc, class Abarrier>
struct MMA_Traits<XE4_ASYNC_GMMA<TD, TC, TA, TB, Shape_MNK_, IsRowMajorA, IsRowMajorB, MatDesc, Abarrier>>
{
  using ValTypeD = TD;
  using ValTypeA = bf16;
  using ValTypeB = bf16;
  using ValTypeC = float;

  using FrgTypeA = xe4::slm_desc<static_cast<int>(!IsRowMajorA), MatDesc>;
  using FrgTypeB = xe4::slm_desc<static_cast<int>(IsRowMajorB), MatDesc>;
  using FrgTypeC = xe4::slm_desc<0, MatDesc>;

  using Shape_MNK = Shape_MNK_;
  using ThrID   = Layout<_1>;
  using ALayout = xe4::ABLayout<get<0>(Shape_MNK{}), get<2>(Shape_MNK{})>;
  using BLayout = xe4::ABLayout<get<1>(Shape_MNK{}), get<2>(Shape_MNK{})>;
  using CLayout = xe4::ABLayout<get<0>(Shape_MNK{}), get<1>(Shape_MNK{})>;

  template<typename... TraitsArgs, __CUTE_REQUIRES(sizeof...(TraitsArgs) <= 4)>
  CUTE_HOST_DEVICE static auto
  with(TraitsArgs&&... args) {
    using MMA_Op = XE4_ASYNC_GMMA<TD, TC, TA, TB, Shape_MNK_, IsRowMajorA, IsRowMajorB, MatDesc, Abarrier>;
    auto opargs = make_tuple(static_cast<TraitsArgs&&>(args)...);
    return MMA_Traits<XE4_ASYNC_GMMA_OP, decltype(opargs), MMA_Op>{{}, opargs};
  }

  template<typename... TraitsArgs, __CUTE_REQUIRES(sizeof...(TraitsArgs) >= 5)>
  CUTE_HOST_DEVICE static auto
  with(TraitsArgs&&... args) {
    using MMA_Op = XE4_ASYNC_GMMA<TD, TC, TA, TB, Shape_MNK_, IsRowMajorA, IsRowMajorB, MatDesc, Abarrier>;
    auto opargs = make_tuple(static_cast<TraitsArgs&&>(args)...);
    auto tmp_opargs = remove<sizeof...(args)-1>(opargs);
    auto opargs_reduced = remove<sizeof...(args)-3>(tmp_opargs);
    return MMA_Traits<XE4_ASYNC_GMMA_OP, decltype(opargs_reduced), MMA_Op>{{}, opargs_reduced};
  }
};

template<typename OpArgs, typename MMA_Op>
struct MMA_Traits<XE4_ASYNC_GMMA_OP, OpArgs, MMA_Op>: public MMA_Traits<MMA_Op> {
  using Abarrier = typename MMA_Op::Abarrier;

  OpArgs const opargs_;

  template <class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE friend constexpr
  void
  mma_unpack(MMA_Traits const& traits,
       Tensor<TD, DLayout>      & D,
       Tensor<TA, ALayout> const& A,
       Tensor<TB, BLayout> const& B,
       Tensor<TC, CLayout> const& C)
  {
    auto matdesc_tuple = make_tuple(*D.data(), *C.data(), *A.data(), *B.data());
    return detail::explode_tuple(detail::CallFMA<MMA_Op>{},
                                 matdesc_tuple, tuple_seq<decltype(matdesc_tuple)>{},
                                 traits.opargs_, tuple_seq<decltype(traits.opargs_)>{});
  }
};

template <class TD, class TC, class TA, class TB, class Shape_MNK_, bool IsRowMajorA, bool IsRowMajorB, class MatDesc, class Abarrier>
struct MMA_Traits<XE4_ASYNC_GMMA_MULTICAST<TD, TC, TA, TB, Shape_MNK_, IsRowMajorA, IsRowMajorB, MatDesc, Abarrier>>
{
  using ValTypeD = TD;
  using ValTypeA = bf16;
  using ValTypeB = bf16;
  using ValTypeC = float;

  using FrgTypeA = xe4::slm_desc<static_cast<int>(!IsRowMajorA), MatDesc>;
  using FrgTypeB = xe4::slm_desc<static_cast<int>(IsRowMajorB), MatDesc>;
  using FrgTypeC = xe4::slm_desc<0, MatDesc>;

  using Shape_MNK = Shape_MNK_;
  using ThrID   = Layout<_1>;
  using ALayout = xe4::ABLayout<get<0>(Shape_MNK{}), get<2>(Shape_MNK{})>;
  using BLayout = xe4::ABLayout<get<1>(Shape_MNK{}), get<2>(Shape_MNK{})>;
  using CLayout = xe4::ABLayout<get<0>(Shape_MNK{}), get<1>(Shape_MNK{})>;

  template<typename... TraitsArgs, __CUTE_REQUIRES(sizeof...(TraitsArgs) <= 4)>
  CUTE_HOST_DEVICE static auto
  with(TraitsArgs&&... args) {
    if constexpr (sizeof...(args) <= 4) {
      return MMA_Traits<XE4_ASYNC_GMMA<TD, TC, TA, TB, Shape_MNK_, IsRowMajorA, IsRowMajorB, MatDesc, Abarrier>>::with(static_cast<TraitsArgs&&>(args)...);
    }
  }

  template<typename... TraitsArgs, __CUTE_REQUIRES(sizeof...(TraitsArgs) > 4)>
  CUTE_HOST_DEVICE static auto
  with(TraitsArgs&&... args) {
    using MMA_Op = XE4_ASYNC_GMMA_MULTICAST<TD, TC, TA, TB, Shape_MNK_, IsRowMajorA, IsRowMajorB, MatDesc, Abarrier>;
    auto opargs = make_tuple(static_cast<TraitsArgs&&>(args)...);
    return MMA_Traits<XE4_ASYNC_GMMA_OP, decltype(opargs), MMA_Op>{{}, opargs};
  }
};

} // namespace cute
