#pragma once

#include "cute/arch/mma_xe4_amma.hpp"
#include "cute/arch/util_xe4.hpp"
#include "util.hpp"

namespace cute::xe4 {

template <xe4::GMMA::Major MajorMode, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr uint32_t get_leading_stride(Layout<Shape, Stride> const&)
{
  constexpr auto ldm = static_cast<int>(MajorMode);
  if constexpr (rank<ldm>(Stride{}) > 1) {
    return get<ldm, 1>(Stride{});
  } else {
    return size(Shape{});
  }
}

template <xe4::GMMA::Major MajorMode, class MatDesc, class SEngine, class SLayout>
CUTE_HOST_DEVICE constexpr
MatDesc make_matrix_desc(Tensor<SEngine, SLayout> const& sTensor) {
  constexpr uint32_t leading_stride = get_leading_stride<MajorMode>(SLayout{});
  constexpr uint32_t cm_stride = (leading_stride * sizeof_bits_v<typename SEngine::value_type> / 8) >> 10;

  MatDesc mat_desc = reinterpret_cast<uint64_t>(slm_space_cast(raw_pointer_cast(sTensor.data()))) >> 9;
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
  print(MatDescIterator const& iter) { printf("xe4::MatDescIterator(%p)", iter.desc_); }
};

template <class T, class MatDesc>
CUTE_HOST_DEVICE constexpr
MatDesc
raw_pointer_cast(MatDescIterator<MatDesc> const& ptr) {
  return ptr.desc_;
}

template <xe4::GMMA::Major MajorMode, class MatDesc>
struct slm_desc : MatDescIterator<MatDesc> { };

template <int M, int K>
using ABLayout = Layout<Shape<_1,Shape<Int<M>,Int<K>>>, Stride<_0,Stride<_1,Int<M>>>>;

} // end namespace cute::xe4

namespace cute {

template <xe4::GMMA::Major MajorMode, class MatDesc>
struct MakeTensor<xe4::slm_desc<MajorMode, MatDesc>>
{
  template <class TEngine, class TLayout>
  CUTE_HOST_DEVICE constexpr auto
  operator()(Tensor<TEngine,TLayout> const& smem_tensor)
  {
    auto mat_desc = xe4::make_matrix_desc<MajorMode, MatDesc>(tensor<0>(smem_tensor));
    auto new_layout = replace<0>(recast<uint8_t const>(smem_tensor).layout(), Layout<_1,_0>{});
    return make_tensor(xe4::MatDescIterator{mat_desc}, new_layout);
  }
};

struct XE4_ASYNC_GMMA_OP {};
struct XE4_ASYNC_GMMA_SCALE_OP {};

template <class TD, class TC, class TA, class TB, class Shape_MNK_, xe4::GMMA::Major tnspA_, xe4::GMMA::Major tnspB_, class MatDesc, class Abarrier>
struct MMA_Traits<XE4_ASYNC_GMMA<TD, TC, TA, TB, Shape_MNK_, tnspA_, tnspB_, MatDesc, Abarrier>>
{
  using ValTypeD = TD;
  using ValTypeA = bf16;
  using ValTypeB = bf16;
  using ValTypeC = float;
  using AbarrierType = Abarrier;

  using FrgTypeA = xe4::slm_desc<tnspA_, MatDesc>;
  using FrgTypeB = xe4::slm_desc<tnspB_, MatDesc>;
  using FrgTypeC = xe4::slm_desc<xe4::GMMA::Major::K, MatDesc>;

  using Shape_MNK = Shape_MNK_;
  using ThrID   = Layout<_1>;
  using ALayout = xe4::ABLayout<get<0>(Shape_MNK{}), get<2>(Shape_MNK{})>;
  using BLayout = xe4::ABLayout<get<1>(Shape_MNK{}), get<2>(Shape_MNK{})>;
  using CLayout = xe4::ABLayout<get<0>(Shape_MNK{}), get<1>(Shape_MNK{})>;

  static constexpr xe4::GMMA::Major tnspA = tnspA_;
  static constexpr xe4::GMMA::Major tnspB = tnspB_;

  template<typename... TraitsArgs, __CUTE_REQUIRES(sizeof...(TraitsArgs) <= 4)>
  CUTE_HOST_DEVICE static auto
  with(TraitsArgs&&... args) {
    using MMA_Op = XE4_ASYNC_GMMA<TD, TC, TA, TB, Shape_MNK_, tnspA_, tnspB_, MatDesc, Abarrier>;
    auto opargs = make_tuple(static_cast<TraitsArgs&&>(args)...);
    return MMA_Traits<XE4_ASYNC_GMMA_OP, decltype(opargs), MMA_Op>{{}, opargs};
  }

  template<typename... TraitsArgs, __CUTE_REQUIRES(sizeof...(TraitsArgs) >= 5)>
  CUTE_HOST_DEVICE static auto
  with(TraitsArgs&&... args) {
    using MMA_Op = XE4_ASYNC_GMMA<TD, TC, TA, TB, Shape_MNK_, tnspA_, tnspB_, MatDesc, Abarrier>;
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

template <class TD, class TC, class TA, class TB, class Shape_MNK_, xe4::GMMA::Major tnspA_, xe4::GMMA::Major tnspB_, class MatDesc, class Abarrier>
struct MMA_Traits<XE4_ASYNC_GMMA_MULTICAST<TD, TC, TA, TB, Shape_MNK_, tnspA_, tnspB_, MatDesc, Abarrier>>
{
  using ValTypeD = TD;
  using ValTypeA = bf16;
  using ValTypeB = bf16;
  using ValTypeC = float;
  using AbarrierType = Abarrier;

  using FrgTypeA = xe4::slm_desc<tnspA_, MatDesc>;
  using FrgTypeB = xe4::slm_desc<tnspB_, MatDesc>;
  using FrgTypeC = xe4::slm_desc<xe4::GMMA::Major::K, MatDesc>;

  using Shape_MNK = Shape_MNK_;
  using ThrID   = Layout<_1>;
  using ALayout = xe4::ABLayout<get<0>(Shape_MNK{}), get<2>(Shape_MNK{})>;
  using BLayout = xe4::ABLayout<get<1>(Shape_MNK{}), get<2>(Shape_MNK{})>;
  using CLayout = xe4::ABLayout<get<0>(Shape_MNK{}), get<1>(Shape_MNK{})>;

  static constexpr xe4::GMMA::Major tnspA = tnspA_;
  static constexpr xe4::GMMA::Major tnspB = tnspB_;

  template<typename... TraitsArgs, __CUTE_REQUIRES(sizeof...(TraitsArgs) <= 4)>
  CUTE_HOST_DEVICE static auto
  with(TraitsArgs&&... args) {
    if constexpr (sizeof...(args) <= 4) {
      return MMA_Traits<XE4_ASYNC_GMMA<TD, TC, TA, TB, Shape_MNK_, tnspA_, tnspB_, MatDesc, Abarrier>>::with(static_cast<TraitsArgs&&>(args)...);
    }
  }

  template<typename... TraitsArgs, __CUTE_REQUIRES(sizeof...(TraitsArgs) > 4)>
  CUTE_HOST_DEVICE static auto
  with(TraitsArgs&&... args) {
    using MMA_Op = XE4_ASYNC_GMMA_MULTICAST<TD, TC, TA, TB, Shape_MNK_, tnspA_, tnspB_, MatDesc, Abarrier>;
    auto opargs = make_tuple(static_cast<TraitsArgs&&>(args)...);
    return MMA_Traits<XE4_ASYNC_GMMA_OP, decltype(opargs), MMA_Op>{{}, opargs};
  }
};

template <class TD, class TC, class TA, class TB, class TMeta, class Shape_MNK_, xe4::GMMA::Major tnspA_, xe4::GMMA::Major tnspB_, bool ScaleA, bool ScaleB, class MatDesc, class MetaDesc, class Abarrier>
struct MMA_Traits<XE4_ASYNC_GMMA_SCALE<TD, TC, TA, TB, TMeta, Shape_MNK_, tnspA_, tnspB_, ScaleA, ScaleB, MatDesc, MetaDesc, Abarrier>>
{
  using ValTypeD = TD;
  using ValTypeA = uint_bit_t<sizeof(TA) * 8 / packed_num<TA>::value>;
  using ValTypeB = uint_bit_t<sizeof(TA) * 8 / packed_num<TB>::value>;
  using ValTypeC = float;
  using ValTypeE = uint_bit_t<sizeof(TA) * 8 / packed_num<TMeta>::value>;

  using FrgTypeA = xe4::slm_desc<tnspA_, MatDesc>;
  using FrgTypeB = xe4::slm_desc<tnspB_, MatDesc>;
  using FrgTypeC = xe4::slm_desc<xe4::GMMA::Major::K, MatDesc>;
  using FrgTypeE = xe4::slm_desc<xe4::GMMA::Major::MN, MetaDesc>;

  using Shape_MNK = Shape_MNK_;
  using ThrID   = Layout<_1>;
  using ALayout = xe4::ABLayout<get<0>(Shape_MNK{}), get<2>(Shape_MNK{})>;
  using BLayout = xe4::ABLayout<get<1>(Shape_MNK{}), get<2>(Shape_MNK{})>;
  using CLayout = xe4::ABLayout<get<0>(Shape_MNK{}), get<1>(Shape_MNK{})>;
  using MetaALayout = ALayout;
  using MetaBLayout = BLayout;

  using AbarrierType = Abarrier;

  static constexpr xe4::GMMA::Major tnspA = tnspA_;
  static constexpr xe4::GMMA::Major tnspB = tnspB_;

  template<typename... TraitsArgs, __CUTE_REQUIRES(sizeof...(TraitsArgs) <= 4)>
  CUTE_HOST_DEVICE static auto
  with(TraitsArgs&&... args) {
    using MMA_Op = XE4_ASYNC_GMMA_SCALE<TD, TC, TA, TB, TMeta, Shape_MNK_, tnspA_, tnspB_, ScaleA, ScaleB, MatDesc, MetaDesc, Abarrier>;
    auto opargs = make_tuple(static_cast<TraitsArgs&&>(args)...);
    return MMA_Traits<XE4_ASYNC_GMMA_SCALE_OP, decltype(opargs), MMA_Op>{{}, opargs};
  }

  template<typename... TraitsArgs, __CUTE_REQUIRES(sizeof...(TraitsArgs) >= 5)>
  CUTE_HOST_DEVICE static auto
  with(TraitsArgs&&... args) {
    using MMA_Op = XE4_ASYNC_GMMA_SCALE<TD, TC, TA, TB, TMeta, Shape_MNK_, tnspA_, tnspB_, ScaleA, ScaleB, MatDesc, MetaDesc, Abarrier>;
    auto opargs = make_tuple(static_cast<TraitsArgs&&>(args)...);
    auto tmp_opargs = remove<sizeof...(args)-1>(opargs);
    auto opargs_reduced = remove<sizeof...(args)-3>(tmp_opargs);
    return MMA_Traits<XE4_ASYNC_GMMA_SCALE_OP, decltype(opargs_reduced), MMA_Op>{{}, opargs_reduced};
  }
};

template<typename OpArgs, typename MMA_Op>
struct MMA_Traits<XE4_ASYNC_GMMA_SCALE_OP, OpArgs, MMA_Op>: public MMA_Traits<MMA_Op> {
  using Abarrier = typename MMA_Op::Abarrier;

  OpArgs opargs_;

  template <class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE friend constexpr
  void
  mma_unpack(MMA_Traits const& traits,
       Tensor<TD, DLayout>      & D,
       Tensor<TA, ALayout> const& A_zipped,
       Tensor<TB, BLayout> const& B_zipped,
       Tensor<TC, CLayout> const& C)
  {
    auto [A, metaA] = unzip_tensor(A_zipped);
    auto [B, metaB] = unzip_tensor(B_zipped);

    auto matdesc_tuple = make_tuple(*D.data(), *C.data(), *A.data(), *B.data(), *metaA.data(), *metaB.data());

    return detail::explode_tuple(detail::CallFMA<MMA_Op>{},
                                 matdesc_tuple, tuple_seq<decltype(matdesc_tuple)>{},
                                 traits.opargs_, tuple_seq<decltype(traits.opargs_)>{});
  }
};

} // namespace cute
