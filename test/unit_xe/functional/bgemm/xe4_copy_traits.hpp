#pragma once

#include "xe4_copy_async.hpp"

#include "cutlass/detail/layout.hpp"

namespace cute
{

struct ASYNC_TENSOR_LOAD_OP : xe4::ASYNC_TENSOR_LOAD {};
struct ASYNC_TENSOR_STORE_OP : xe4::ASYNC_TENSOR_STORE {};

template <class CopyOp>
struct XE4_COPY_Unpack
{
  template <class... Args,
            class TS, class SLayout,
            class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits<CopyOp, Args...> const& traits,
              Tensor<TS,SLayout>           const& src,
              Tensor<TD,DLayout>                & dst)
  {
    if constexpr (is_same_v<CopyOp, ASYNC_TENSOR_LOAD_OP>) {
      auto dst_ptr = dst.data();
      auto src_coord = src.data().coord_;
      return detail::explode_tuple(detail::CallCOPY<CopyOp>{},
                                  traits.opargs_, tuple_seq<decltype(traits.opargs_)>{},
                                  make_tuple(dst_ptr), seq<0>{}, src_coord, tuple_seq<decltype(src_coord)>{});
    } else {
      auto src_ptr = src.data();
      auto dst_coord = dst.data().coord_;
      return detail::explode_tuple(detail::CallCOPY<CopyOp>{},
                                  traits.opargs_, tuple_seq<decltype(traits.opargs_)>{},
                                  make_tuple(src_ptr), seq<0>{}, dst_coord, tuple_seq<decltype(dst_coord)>{});
    }
  }
};

template <class CopyOperation, class NumBitsPerTMA, class TensorDesc, class GBasis>
struct Copy_Traits<CopyOperation, NumBitsPerTMA, TensorDesc, GBasis>
{
  using ThrID     = Layout<_1>;
  using SrcLayout = Layout<Shape<_1, NumBitsPerTMA>>;
  using DstLayout = SrcLayout;
  using RefLayout = SrcLayout;

  using CopyOp = conditional_t<is_same_v<CopyOperation, xe4::ASYNC_TENSOR_LOAD>, ASYNC_TENSOR_LOAD_OP, ASYNC_TENSOR_STORE_OP>;

  TensorDesc* tdesc_ptr_ = nullptr;
  GBasis gbasis_;

  template<class ABarrier>
  CUTE_HOST_DEVICE constexpr
  Copy_Traits<CopyOp, NumBitsPerTMA, TensorDesc, ABarrier, bool>
  with(ABarrier const* abar_ptr) const {
    return {{}, {tdesc_ptr_, abar_ptr}};
  }

  template <class GShape>
  CUTE_HOST_DEVICE constexpr
  auto get_tma_tensor(GShape const& g_shape) const {
    return make_counting_tensor(make_layout(g_shape, gbasis_));
  }

  // Don't try to execute a copy with XE4_TMA_LOAD before calling .with()
  template <class TS, class SLayout,
            class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits        const& traits,
              Tensor<TS,SLayout> const& src,
              Tensor<TD,DLayout>      & dst) = delete;
};

template<class CopyOp, class NumBitsPerTMA, class TensorDesc, class GBasis>
auto make_copy_traits(TensorDesc* tensor_desc, GBasis gbasis)
{
  return Copy_Traits<CopyOp, NumBitsPerTMA, TensorDesc, GBasis>{tensor_desc, gbasis};
}

template <class CopyOp, class NumBitsPerTMA, class TensorDesc, class ABarrier, class Unused>
struct Copy_Traits<CopyOp, NumBitsPerTMA, TensorDesc, ABarrier, Unused> : XE4_COPY_Unpack<CopyOp>
{
  using ThrID     = Layout<_1>;
  using SrcLayout = Layout<Shape<_1, NumBitsPerTMA>>;
  using DstLayout = SrcLayout;
  using RefLayout = SrcLayout;

  tuple<TensorDesc const*, ABarrier const*> const opargs_;
};

namespace detail {

template<cm_size_t cmSize_, cm_layout_t cmLayout_, bool swizzlingDisabled_, class tdescPtr_, int tdescIdx_>
struct AuxParams {
  using tdescPtr = tdescPtr_;
  static constexpr int tdescIdx = tdescIdx_;
  static constexpr cm_size_t cmSize = cmSize_;
  static constexpr cm_layout_t cmLayout = cmLayout_;
  static constexpr bool swizzlingDisabled = swizzlingDisabled_;
};

template <class AuxParams, bool isTransposed, class GTensor, class SLayout>
CUTE_HOST_DEVICE auto
make_tensor_desc(GTensor const& gtensor, SLayout const& slayout)
{
  using T = typename GTensor::value_type;

  constexpr int dim_x = isTransposed ? 0 : 1;
  constexpr int dim_y = dim_x ^ 1;

  uint32_t width = size<dim_x>(gtensor);
  uint32_t height = size<dim_y>(gtensor);
  uint32_t block_width = size<dim_x>(slayout);
  uint32_t block_height = size<dim_y>(slayout);

  auto tdesc_ptr = allocate_tdesc<AuxParams::tdescIdx, typename AuxParams::tdescPtr>();
  tensor_desc_fill_global_addr(tdesc_ptr, gtensor.data());
  tensor_descriptor_fill_dim_size<2>(tdesc_ptr, {width, height});
  tensor_descriptor_fill_dim_stride<2>(tdesc_ptr, width * sizeof(T));
  tensor_descriptor_fill_traverse_stride<2>(tdesc_ptr, sycl::vec<uint32_t, 2>{1, 1});
  tensor_descriptor_fill_roitensor_size<2>(tdesc_ptr, {block_width, block_height});
  tensor_descriptor_fill_misc<T, AuxParams::cmSize, AuxParams::cmLayout, AuxParams::swizzlingDisabled>(tdesc_ptr);

  return tdesc_ptr;
}

template <class Shape, class Stride>
constexpr bool
is_mn_major(Layout<Shape,Stride> const& layout) {
  return cutlass::detail::is_major<0, Stride>();
}

template <bool isTransposed>
CUTE_HOST_DEVICE auto get_gbasis() {
  if constexpr (isTransposed) {
    return make_stride(E<1>{}, E<0>{}, E<2>{});
  } else {
    return make_stride(E<0>{}, E<1>{}, E<2>{});
  }
}

template <class CopyOp, class AuxParams, class GEngine, class GLayout, class SLayout>
CUTE_HOST_DEVICE auto
make_copy_atom(Tensor<GEngine, GLayout> const& gtensor, SLayout const& slayout)
{
  using T = typename GEngine::value_type;

  auto num_elems_per_tma = size<0>(group<0, 2>(slayout));
  constexpr uint32_t num_bits_per_tma = num_elems_per_tma * sizeof_bits_v<T>;

  constexpr bool isTransposed = is_mn_major(GLayout{});
  auto gbasis = get_gbasis<isTransposed>();
  auto tensor_desc= make_tensor_desc<AuxParams, isTransposed>(gtensor, slayout);
  auto copy_traits = make_copy_traits<CopyOp, C<num_bits_per_tma>>(tensor_desc, gbasis);
  return Copy_Atom<decltype(copy_traits), T>{copy_traits};
}

template <class CopyOp, class AuxParams, class GTensor, class SLayout, class TLayout, class VLayout>
CUTE_HOST_RTC auto
make_xe4_copy_tiled(GTensor const& gtensor, SLayout const& slayout,
                    TLayout const& cta_t_map, VLayout const& cta_v_map)
{
  auto cta_tiler = product_each(shape(cta_v_map));
  auto atom = make_copy_atom<CopyOp,AuxParams>(gtensor, slayout);

  auto num_elems_per_tma = size<1>(typename decltype(atom)::RefLayout{}) / static_value<sizeof_bits<typename GTensor::value_type>>();

  // smem idx -> smem coord
  auto inv_smem_layout = right_inverse(get_nonswizzle_portion(slayout));
  // CTA V -> smem_coord
  auto layout_v = composition(inv_smem_layout, num_elems_per_tma);
  // Scale that up to cover all of the smem_coords
  auto layout_V = tile_to_shape(make_layout(layout_v), size(cta_v_map));
  // CTA T -> smem idx
  auto layout_t = make_layout(cosize(cta_t_map), shape_div(num_elems_per_tma, cosize(cta_t_map)));
  // CTA TID -> smem coord
  auto layout_T = composition(inv_smem_layout, composition(layout_t, cta_t_map));
  // Combine with the T mapping
  auto layout_TV = make_layout(layout_T, layout_V);

  return TiledCopy<decltype(atom), decltype(layout_TV), decltype(cta_tiler)>{atom};
}

template <class CopyOp, class AuxParams, class GTensor, class SLayout, class Tiler>
CUTE_HOST_RTC auto
make_xe4_copy(GTensor const& gtensor, SLayout const& slayout, Tiler const& cta_tiler)
{
  auto cluster_size = _1{};
  auto cta_t_tile = make_layout(cluster_size);
  auto cta_v_tile = make_identity_layout(shape(gtensor)).compose(cta_tiler);
  return make_xe4_copy_tiled<CopyOp,AuxParams>(gtensor, slayout, cta_t_tile, cta_v_tile);
}

} // namespace detail

} // namespace cute
