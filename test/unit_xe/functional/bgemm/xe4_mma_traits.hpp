#pragma once

#include "xe4_mma.hpp"
#include "util.hpp"

namespace cute::xe4 {

template <uint32_t Height, uint32_t WidthInBytes, class MatDesc=uint64_t, class SEngine, class SLayout>
CUTE_HOST_DEVICE constexpr
MatDesc make_mat_desc(Tensor<SEngine,SLayout> const& sTensor) {
  using T = typename SEngine::value_type;
  constexpr int leading_dim = is_mn_major(SLayout{}) ? 0 : 1;

  constexpr uint32_t width = WidthInBytes / sizeof(T);
  constexpr uint32_t x_iteration = size<leading_dim>(SLayout{}) / width;
  constexpr uint64_t row_offset = (sizeof(T) * width * Height * x_iteration) >> 9;
  constexpr uint64_t col_offset = (width * Height * sizeof(T)) >> 9;

  MatDesc mat_desc = reinterpret_cast<uint64_t>(slm_space_cast(sTensor.data())) >> 9;
  mat_desc |= (row_offset << 16) | (col_offset << 32);

  return mat_desc;
}

template <cm_size_t cmSize, class MatDesc=uint64_t, class STensor>
CUTE_HOST_DEVICE constexpr
MatDesc make_mat_desc(STensor const& sTensor) {
  constexpr uint32_t Height = get_height<cmSize>();
  constexpr uint32_t WidthInBytes = get_width_in_bytes<cmSize>();
  return make_mat_desc<Height, WidthInBytes, MatDesc>(sTensor);
}

template <class MatDesc=uint64_t>
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
    return { MatDesc{desc_ + (uint64_t(offset) >> 9)} };
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

template <cm_size_t cmSize_, class MatDesc=uint64_t>
struct slm_desc : MatDescIterator<MatDesc> {
  static constexpr cm_size_t cmSize = cmSize_;
};

template <int M, int K>
using ABLayout = Layout<Shape<_1,Shape<Int<M>,Int<K>>>, Stride<_0,Stride<_1,Int<M>>>>;

} // end namespace cute::xe4

namespace cute {

template <cm_size_t cmSize, class MatDesc>
struct MakeTensor<xe4::slm_desc<cmSize, MatDesc>>
{
  template <class TEngine, class TLayout>
  CUTE_HOST_DEVICE constexpr auto
  operator()(Tensor<TEngine,TLayout> const& smem_tensor)
  {
    auto mat_desc = xe4::make_mat_desc<cmSize, MatDesc>(tensor<0>(smem_tensor));
    auto new_layout = replace<0>(recast<uint8_t const>(smem_tensor).layout(), Layout<_1,_0>{});
    return make_tensor(xe4::MatDescIterator{mat_desc}, new_layout);
  }
};

struct XE4_ASYNC_GMMA_OP {};

template <class TD, class TC, class TA, class TB, class Shape_MNK_, class CoreMatSize, class MatDesc, class Abarrier>
struct MMA_Traits<XE4_ASYNC_GMMA<TD, TC, TA, TB, Shape_MNK_, CoreMatSize, MatDesc, Abarrier>>
{
  using ValTypeD = float;
  using ValTypeA = bf16;
  using ValTypeB = bf16;
  using ValTypeC = float;

  using FrgTypeA = xe4::slm_desc<CoreMatSize::cmSizeA>;
  using FrgTypeB = xe4::slm_desc<CoreMatSize::cmSizeB>;
  using FrgTypeC = xe4::slm_desc<CoreMatSize::cmSizeC>;

  using Shape_MNK = Shape_MNK_;
  using ThrID   = Layout<_1>;
  using ALayout = xe4::ABLayout<get<0>(Shape_MNK{}), get<2>(Shape_MNK{})>;
  using BLayout = xe4::ABLayout<get<1>(Shape_MNK{}), get<2>(Shape_MNK{})>;
  using CLayout = xe4::ABLayout<get<0>(Shape_MNK{}), get<1>(Shape_MNK{})>;

  template<typename MMA_Op>
  CUTE_HOST_DEVICE auto
  with(Abarrier const& abarrier, MMA_Op &&) const {
    return MMA_Traits<XE4_ASYNC_GMMA_OP, MMA_Op>{abarrier};
  }
};

template<typename MMA_Op>
struct MMA_Traits<XE4_ASYNC_GMMA_OP, MMA_Op>: public MMA_Traits<MMA_Op> {
  using Abarrier = typename MMA_Op::Abarrier;

  const Abarrier abarrier_;

  CUTE_HOST_DEVICE MMA_Traits(Abarrier const& abarrier) : abarrier_(abarrier) {}

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
    return detail::explode_tuple(detail::CallFMA<MMA_Op>{},
                                 make_tuple(traits.abarrier_, *D.data(), *C.data(), *A.data(), *B.data()), seq<0,1,2,3,4>{});
  }
};

} // namespace cute
