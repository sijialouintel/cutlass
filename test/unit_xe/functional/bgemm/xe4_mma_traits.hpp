#pragma once

#include "xe4_mma.hpp"
#include "util.hpp"

namespace cute::xe4 {

enum class MatrixTag : uint32_t {
  A = 0,
  B = 1,
  C = 2
};

template <bool isRowMajor, class MatDesc, class Tensor>
CUTE_HOST_DEVICE constexpr
MatDesc make_mat_desc_a(Tensor const& sTensor) {
  using T = typename Tensor::value_type;
  constexpr int leading_dim = isRowMajor ? 1 : 0;

  constexpr uint32_t cm_bytes = 1024;
  constexpr uint32_t cm_size_a_x = isRowMajor ? (32 / sizeof(T)) : 32;
  constexpr uint32_t cm_num_a_x = size<leading_dim>(typename Tensor::layout_type{}) / cm_size_a_x;
  constexpr uint32_t cm_stride_a = (cm_bytes * cm_num_a_x) >> 10;

  MatDesc mat_desc = reinterpret_cast<uint64_t>(slm_space_cast(sTensor.data())) >> 9;
  mat_desc |= (cm_stride_a << 16);

  return mat_desc;
}

template <bool isRowMajor, class MatDesc, class Tensor>
CUTE_HOST_DEVICE constexpr
MatDesc make_mat_desc_b(Tensor const& sTensor) {
  using T = typename Tensor::value_type;
  constexpr int leading_dim = isRowMajor ? 0 : 1;

  constexpr uint32_t cm_bytes = 1024;
  constexpr uint32_t cm_size_b_x = 32 / sizeof(T);
  constexpr uint32_t cm_num_b_x = size<leading_dim>(typename Tensor::layout_type{}) / cm_size_b_x;
  constexpr uint32_t cm_stride_b = (cm_bytes * cm_num_b_x) >> 10;

  MatDesc mat_desc = reinterpret_cast<uint64_t>(slm_space_cast(sTensor.data())) >> 9;
  mat_desc |= (cm_stride_b << 16);

  return mat_desc;
}

template <bool isRowMajor, class MatDesc, class Tensor>
CUTE_HOST_DEVICE constexpr
MatDesc make_mat_desc_c(Tensor const& sTensor) {
  using T = typename Tensor::value_type;
  constexpr int leading_dim = isRowMajor ? 1 : 0;

  constexpr uint32_t cm_bytes = 1024;
  constexpr uint32_t cm_size_c_x = 32 / sizeof(T);
  constexpr uint32_t cm_num_c_x = size<1>(typename Tensor::layout_type{}) / cm_size_c_x;
  constexpr uint32_t cm_stride_c = (cm_bytes * cm_num_c_x) >> 10;

  MatDesc mat_desc = reinterpret_cast<uint64_t>(slm_space_cast(sTensor.data())) >> 9;
  mat_desc |= (cm_stride_c << 16);

  return mat_desc;
}

template <MatrixTag matrixTag, bool isRowMajor, class MatDesc, class STensor>
CUTE_HOST_DEVICE constexpr
MatDesc make_mat_desc(STensor const& sTensor) {
  if constexpr (matrixTag == MatrixTag::A) {
    return make_mat_desc_a<isRowMajor, MatDesc>(sTensor);
  } else if constexpr (matrixTag == MatrixTag::B) {
    return make_mat_desc_b<isRowMajor, MatDesc>(sTensor);
  } else {
    return make_mat_desc_c<isRowMajor, MatDesc>(sTensor);
  }
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

template <MatrixTag matrixTag, bool isRowMajor, class MatDesc>
struct slm_desc : MatDescIterator<MatDesc> { };

template <int M, int K>
using ABLayout = Layout<Shape<_1,Shape<Int<M>,Int<K>>>, Stride<_0,Stride<_1,Int<M>>>>;

} // end namespace cute::xe4

namespace cute {

template <xe4::MatrixTag matrixTag, bool isRowMajor, class MatDesc>
struct MakeTensor<xe4::slm_desc<matrixTag, isRowMajor, MatDesc>>
{
  template <class TEngine, class TLayout>
  CUTE_HOST_DEVICE constexpr auto
  operator()(Tensor<TEngine,TLayout> const& smem_tensor)
  {
    auto mat_desc = xe4::make_mat_desc<matrixTag, isRowMajor, MatDesc>(tensor<0>(smem_tensor));
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

  using FrgTypeA = xe4::slm_desc<xe4::MatrixTag::A, IsRowMajorA, MatDesc>;
  using FrgTypeB = xe4::slm_desc<xe4::MatrixTag::B, IsRowMajorB, MatDesc>;
  using FrgTypeC = xe4::slm_desc<xe4::MatrixTag::C, true, MatDesc>;

  using Shape_MNK = Shape_MNK_;
  using ThrID   = Layout<_1>;
  using ALayout = xe4::ABLayout<get<0>(Shape_MNK{}), get<2>(Shape_MNK{})>;
  using BLayout = xe4::ABLayout<get<1>(Shape_MNK{}), get<2>(Shape_MNK{})>;
  using CLayout = xe4::ABLayout<get<0>(Shape_MNK{}), get<1>(Shape_MNK{})>;

  template<typename MMA_Op, class... TraitsArgs>
  CUTE_HOST_DEVICE auto
  with(MMA_Op && mma_op, Abarrier const& abarrier, [[maybe_unused]] TraitsArgs&&... args) const {
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


struct XE4_ASYNC_GMMA_MULTICAST_OP {};

template <class TD, class TC, class TA, class TB, class Shape_MNK_, bool IsRowMajorA, bool IsRowMajorB, class MatDesc, class Abarrier>
struct MMA_Traits<XE4_ASYNC_GMMA_MULTICAST<TD, TC, TA, TB, Shape_MNK_, IsRowMajorA, IsRowMajorB, MatDesc, Abarrier>>
{
  using ValTypeD = TD;
  using ValTypeA = bf16;
  using ValTypeB = bf16;
  using ValTypeC = float;

  using FrgTypeA = xe4::slm_desc<xe4::MatrixTag::A, IsRowMajorA, MatDesc>;
  using FrgTypeB = xe4::slm_desc<xe4::MatrixTag::B, IsRowMajorB, MatDesc>;
  using FrgTypeC = xe4::slm_desc<xe4::MatrixTag::C, true, MatDesc>;

  using Shape_MNK = Shape_MNK_;
  using ThrID   = Layout<_1>;
  using ALayout = xe4::ABLayout<get<0>(Shape_MNK{}), get<2>(Shape_MNK{})>;
  using BLayout = xe4::ABLayout<get<1>(Shape_MNK{}), get<2>(Shape_MNK{})>;
  using CLayout = xe4::ABLayout<get<0>(Shape_MNK{}), get<1>(Shape_MNK{})>;

  template<typename MMA_Op, class... TraitsArgs>
  CUTE_HOST_DEVICE auto
  with(MMA_Op && mma_op, TraitsArgs&&... args) const {
    return MMA_Traits<XE4_ASYNC_GMMA_MULTICAST_OP, MMA_Op>{{}, {static_cast<TraitsArgs&&>(args)...}};
  }
};

template<typename MMA_Op>
struct MMA_Traits<XE4_ASYNC_GMMA_MULTICAST_OP, MMA_Op>: public MMA_Traits<MMA_Op> {
  using Abarrier = typename MMA_Op::Abarrier;

  tuple<Abarrier, uint32_t, uint32_t> const opargs_;

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
                                 traits.opargs_, tuple_seq<decltype(traits.opargs_)>{},
                                 make_tuple(*D.data(), *C.data(), *A.data(), *B.data()), seq<0,1,2,3>{});
  }
};

} // namespace cute
