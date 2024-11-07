#pragma once

#include "cute/arch/copy_xe4_dma.hpp"
#include "cutlass/detail/layout.hpp"
#include "cute/atom/copy_traits_sm90_tma.hpp"

namespace cute
{

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
    constexpr auto isLoadOperation = !cute::is_base_of<xe4::ASYNC_TENSOR_STORE, CopyOp>::value;

    if constexpr (isLoadOperation) {
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

template <class CopyOp>
struct SLM_VCOPY_Unpack
{
  template <class... Args,
            class TS, class SLayout,
            class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits<CopyOp, Args...> const& traits,
              Tensor<TS,SLayout>           const& src,
              Tensor<TD,DLayout>                & dst)
  {
    return detail::explode_tuple(detail::CallCOPY<CopyOp>{}, make_tuple(src.data(), dst.data()), seq<0,1>{});
  }
};


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////// ASYNC_TENSOR_LOAD / ASYNC_TENSOR_STORE ///////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename CopyOperation>
struct Xe4CopyOp {};

template <typename CopyOperation>
struct Xe4CopyOpWrapper : CopyOperation {};

template <class TensorDesc, class AuxParams>
struct Xe4DmaCache {
  template <typename CopyOp>
  using OpUnpack = XE4_COPY_Unpack<CopyOp>;

  CUTE_HOST_DEVICE constexpr
  auto get_tensor_desc() const {
    return tdesc_ptr_;
  }

  template <class GShape>
  CUTE_HOST_DEVICE constexpr
  auto get_tma_tensor(GShape const& g_shape) const {
    static_assert(is_congruent<decltype(g_shape), decltype(aux_params_.g_stride_)>::value);
    return make_counting_tensor(make_layout(g_shape, aux_params_.g_stride_));
  }

  TensorDesc tdesc_ptr_;
  AuxParams aux_params_;
};

template <class CopyOperation, class NumBitsPerTMA, class DmaCache>
struct Copy_Traits<Xe4CopyOp<CopyOperation>, NumBitsPerTMA, DmaCache>
{
  using ThrID     = Layout<_1>;
  using SrcLayout = Layout<Shape<_1, NumBitsPerTMA>>;
  using DstLayout = SrcLayout;
  using RefLayout = SrcLayout;

  DmaCache cache_;

  template<class ABarrier>
  CUTE_HOST_DEVICE constexpr
  auto with(ABarrier const* abar_ptr, [[maybe_unused]] uint32_t const& multicast_mask = 0) const {
    using Wrapper = Xe4CopyOpWrapper<CopyOperation>;
    using OpUnpack = typename DmaCache::template OpUnpack<Wrapper>;

    auto opargs = make_tuple(cache_.get_tensor_desc(), abar_ptr);
    return Copy_Traits<Wrapper, NumBitsPerTMA, decltype(opargs), OpUnpack>{opargs};
  }

  template <class GShape>
  CUTE_HOST_DEVICE constexpr
  auto get_tma_tensor(GShape const& g_shape) const {
    return cache_.get_tma_tensor(g_shape);
  }

  // Don't try to execute a copy with XE4_TMA_LOAD before calling .with()
  template <class TS, class SLayout,
            class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits        const& traits,
              Tensor<TS,SLayout> const& src,
              Tensor<TD,DLayout>      & dst) = delete;
};

template <class CopyOperation, class NumBitsPerTMA, class OpArgsTuple, template<class> class OpUnpack>
struct Copy_Traits<Xe4CopyOpWrapper<CopyOperation>, NumBitsPerTMA, OpArgsTuple, OpUnpack<Xe4CopyOpWrapper<CopyOperation>>> : OpUnpack<Xe4CopyOpWrapper<CopyOperation>>
{
  using ThrID     = Layout<_1>;
  using SrcLayout = Layout<Shape<_1, NumBitsPerTMA>>;
  using DstLayout = SrcLayout;
  using RefLayout = SrcLayout;

  OpArgsTuple const opargs_;

  Copy_Traits(OpArgsTuple const& opargs) : opargs_(opargs) {}
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////// ASYNC_TENSOR_LOAD_MULTICAST / ASYNC_TENSOR_STORE_MULTICAST /////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <class NumBitsPerTMA, class DmaCache>
struct Copy_Traits<Xe4CopyOp<xe4::ASYNC_TENSOR_LOAD_MULTICAST>, NumBitsPerTMA, DmaCache>
{
  using ThrID     = Layout<_1>;
  using SrcLayout = Layout<Shape<_1, NumBitsPerTMA>>;
  using DstLayout = SrcLayout;
  using RefLayout = SrcLayout;

  DmaCache cache_;

  template<class ABarrier>
  CUTE_HOST_DEVICE constexpr
  auto with(ABarrier const* abar_ptr, uint32_t const& multicast_mask) const {
    using CopyOperation = xe4::ASYNC_TENSOR_LOAD_MULTICAST;
    using Wrapper = Xe4CopyOpWrapper<CopyOperation>;
    using OpUnpack = typename DmaCache::template OpUnpack<Wrapper>;

    auto opargs = make_tuple(cache_.get_tensor_desc(), abar_ptr, multicast_mask);
    return Copy_Traits<Wrapper, NumBitsPerTMA, decltype(opargs), OpUnpack>{opargs};
  }

  template <class GShape>
  CUTE_HOST_DEVICE constexpr
  auto get_tma_tensor(GShape const& g_shape) const {
    return cache_.get_tma_tensor(g_shape);
  }

  // Don't try to execute a copy with XE4_TMA_LOAD before calling .with()
  template <class TS, class SLayout,
            class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits        const& traits,
              Tensor<TS,SLayout> const& src,
              Tensor<TD,DLayout>      & dst) = delete;
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////// SLM_VLOAD / SLM_VSTORE ///////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<uint32_t VS, class NumBitsPerCopy>
struct Copy_Traits<xe4::SLM_VLOAD<VS>, NumBitsPerCopy> : SLM_VCOPY_Unpack<xe4::SLM_VLOAD<VS>>
{
  using ThrID     = Layout<_1>;
  using SrcLayout = Layout<Shape<_1,NumBitsPerCopy>>;
  using DstLayout = SrcLayout;
  using RefLayout = SrcLayout;
};

template<uint32_t VS, class NumBitsPerCopy>
struct Copy_Traits<xe4::SLM_VSTORE<VS>, NumBitsPerCopy> : SLM_VCOPY_Unpack<xe4::SLM_VSTORE<VS>>
{
  using ThrID     = Layout<_1>;
  using SrcLayout = Layout<Shape<_1,NumBitsPerCopy>>;
  using DstLayout = SrcLayout;
  using RefLayout = SrcLayout;
};

namespace detail {
template <class TmaInternalType,
          class GEngine, class GLayout,
          class TShape, class TStride,
          int B, int M, int S>
CUTE_HOST_RTC
auto
make_tma_copy_aux_params(Tensor<GEngine,GLayout> const& gtensor,         // The original GMEM Tensor
                   Layout<TShape,TStride>  const& tma_gbasis,      // TMA mode -> GMEM mode mapping
                   Swizzle<B,M,S>          const& swizzle)         // Swizzle fn on smem_idx
{
  // Recast the original tensor for shape/stride inspections
  Tensor gtensor_T = recast<TmaInternalType>(gtensor);
  auto  gmem_layout  = gtensor_T.layout();

  cute::array<uint64_t, 5> gmem_prob_shape  = {1,1,1,1,1};
  cute::array<uint64_t, 5> gmem_prob_stride = {0,0,0,0,0};
  fill_tma_gmem_shape_stride(gtensor_T, stride(tma_gbasis), gmem_prob_shape, gmem_prob_stride);

  auto recast_ratio = cute::trait_ratio(sizeof_bits<typename GEngine::value_type>{},
                                        sizeof_bits<             TmaInternalType>{});

  auto gbasis = make_basis_like(shape(gtensor));

  // Finally, get the inverse permutation of the E<i> bases for the mocked gmem stride
  auto gmem_tma_basis_stride = transform_leaf(gbasis, [&](auto ei) {
    auto si = basis_get(ei,  shape(gmem_layout));
    auto di = basis_get(ei, stride(gmem_layout));
    if constexpr (is_constant<1, decltype(si)>::value || is_constant<0, decltype(di)>::value) {
      return Int<0>{};                  // If size-1 or stride-0, return arithmetic identity -- no contribution to the TMA
    } else {
      auto tma_gmem_basis_stride = stride(tma_gbasis);
      // Find j such that E<i> is in stride<j>(tma_gbasis)
      using EI = decltype(ei);
      [[maybe_unused]] auto j = find_if(tma_gmem_basis_stride, [&](auto tma_stride_j) { return any_of(tma_stride_j, [&](auto dj) { return dj == EI{}; }); });
      if constexpr (decltype(j == rank(tma_gmem_basis_stride))::value) {
        return Int<0>{};               // If not-found, return arithmetic identity -- no contribution to the TMA
      } else
      if constexpr (decltype(j == Int<0>{})::value) {
        auto scale = recast_ratio * basis_get(ei, stride(gtensor));
        return E<j>{} * scale;         // Return TMA Coord basis -- with a recast scale factor
      } else
      if constexpr (decltype(rank<j>(tma_gmem_basis_stride) == Int<1>{})::value) {
        return E<j>{};                 // Return TMA Coord basis -- known scale of Int<1>{}
      } else {
        int32_t scale = ceil_div(int32_t(di * sizeof_bits_v<TmaInternalType> / cute::max(gmem_prob_stride[j], uint64_t{16})), 8);
        return E<j>{} * scale;         // Return TMA Coord basis -- with a dynamic scale factor
      }
    }
  });

#if 0
    print("gmem_tma_basis_stride : "); print(gmem_tma_basis_stride); print("\n");
#endif

  using AuxParams = AuxTmaParams<decltype(gmem_tma_basis_stride),
                                 decltype(tma_gbasis),
                                 decltype(swizzle)>;
  return AuxParams{gmem_tma_basis_stride};
}

template<slm_matrix_type cmType_, class tdescPtr_, int tdescIdx_>
struct AuxParams {
  using tdescPtr = tdescPtr_;
  static constexpr int tdescIdx = tdescIdx_;
  static constexpr slm_matrix_type cmType = cmType_;
};

template <class Shape, class Stride>
constexpr int get_leading_dim(Layout<Shape,Stride> const& layout) {
  bool is_k_major = cutlass::detail::is_major<1, Stride>();
  return static_cast<int>(is_k_major);
}

template <class AuxParams, class TmaInternalType, class GEngine, class GLayout, class SLayout>
CUTE_HOST_DEVICE auto
make_tensor_desc(Tensor<GEngine, GLayout> const& gtensor, SLayout const& slayout, uint32_t coop_size)
{
  constexpr int ldm = get_leading_dim(GLayout{});
  constexpr int non_ldm = ldm ^ 1;

  // Recast the original tensor for shape/stride inspections
  Tensor gtensor_T = recast<TmaInternalType>(gtensor);

  void* gmem_address = (void*) raw_pointer_cast(gtensor_T.data());
  auto  gmem_layout  = gtensor_T.layout();

  uint32_t width = size<ldm>(gmem_layout);
  uint32_t height = size<non_ldm>(gmem_layout);
  uint32_t block_width = size<ldm>(slayout);
  uint32_t block_height = size<non_ldm>(slayout) / coop_size;

  auto tdesc_ptr = allocate_tdesc<AuxParams::tdescIdx, typename AuxParams::tdescPtr>();
  tensor_desc_fill_global_addr(tdesc_ptr, gmem_address);
  tensor_descriptor_fill_dim_size<2>(tdesc_ptr, {width, height});
  tensor_descriptor_fill_dim_stride<2>(tdesc_ptr, width * sizeof(TmaInternalType));
  tensor_descriptor_fill_traverse_stride<2>(tdesc_ptr, sycl::vec<uint32_t, 2>{1, 1});
  tensor_descriptor_fill_roitensor_size<2>(tdesc_ptr, {block_width, block_height});
  tensor_descriptor_fill_misc<typename GEngine::value_type, AuxParams::cmType>(tdesc_ptr);

  return tdesc_ptr;
}

template <class AuxParams, class GTensor, class SLayout>
CUTE_HOST_DEVICE auto
make_conv2d_tensor_desc(GTensor const& gtensor, SLayout const& slayout, uint32_t coop_size = 1)
{
  using T = typename GTensor::value_type;

  // Note: adjust the shape order to align with that defined in cutlass::conv::ConvProblemShape
  sycl::vec<uint32_t, 4> gmem_shape {(uint32_t)shape<1, 0>(gtensor),
    (uint32_t)shape<1, 1>(gtensor), (uint32_t)shape<1, 2>(gtensor), (uint32_t)shape<0>(gtensor)};
  sycl::vec<uint64_t, 3> gmem_stride {stride<1, 1>(gtensor) * sizeof(T), stride<1, 2>(gtensor) * sizeof(T),
    stride<0>(gtensor) * sizeof(T)};
  sycl::vec<uint32_t, 4> roi_shape {shape<1>(slayout), 1, 1, shape<0>(slayout) / coop_size};
  sycl::vec<uint32_t, 4> elem_stride {1, 1, 1, 1};

  auto tdesc_ptr = allocate_tdesc<AuxParams::tdescIdx, typename AuxParams::tdescPtr>();
  tensor_desc_fill_global_addr(tdesc_ptr, gtensor.data());
  tensor_descriptor_fill_dim_size<4>(tdesc_ptr, gmem_shape);
  tensor_descriptor_fill_dim_stride<4>(tdesc_ptr, gmem_stride);
  tensor_descriptor_fill_traverse_stride<4>(tdesc_ptr, elem_stride);
  tensor_descriptor_fill_roitensor_size<4>(tdesc_ptr, roi_shape);
  tensor_descriptor_fill_misc<T, AuxParams::cmType>(tdesc_ptr);

  return tdesc_ptr;
}

template <class CopyOp, class AuxParams, class TmaInternalType, class GEngine, class GLayout, class SLayout, class VLayout>
CUTE_HOST_DEVICE auto
make_copy_atom(Tensor<GEngine, GLayout> const& gtensor, SLayout const& slayout, uint32_t coop_size, VLayout const& cta_v_map)
{
  using T = typename GEngine::value_type;

  auto num_elems_per_tma = size<0>(group<0, 2>(slayout));
  constexpr uint32_t num_bits_per_tma = num_elems_per_tma * sizeof_bits_v<T>;

  auto smem_swizzle = get_swizzle_portion(slayout);
  auto smem_layout  = get_nonswizzle_portion(slayout);

  auto tma_gbasis = detail::construct_tma_gbasis<TmaInternalType>(gtensor, slayout, cta_v_map);

  auto tensor_desc = make_tensor_desc<AuxParams, TmaInternalType>(gtensor, slayout, coop_size);
  auto aux_params = make_tma_copy_aux_params<TmaInternalType>(gtensor, tma_gbasis, smem_swizzle);

  using DmaCache = Xe4DmaCache<decltype(tensor_desc), decltype(aux_params)>;
  using Traits = Copy_Traits<Xe4CopyOp<CopyOp>, cute::C<num_bits_per_tma>, DmaCache>;
  using Atom   = Copy_Atom<Traits, typename GEngine::value_type>;

  Traits tma_traits{{tensor_desc, aux_params}};

  // Return the Copy_Atom
  return Atom{tma_traits};
}

template <class CopyOp, class AuxParams, class GEngine, class GLayout, class SLayout, class VLayout>
CUTE_HOST_DEVICE auto
make_copy_atom_conv2d(Tensor<GEngine, GLayout> const& gtensor, SLayout const& slayout,
uint32_t coop_size, VLayout const& cta_v_map)
{
  using T = typename GEngine::value_type;

  auto num_elems_per_tma = size<0>(group<0, 2>(slayout));
  constexpr uint32_t num_bits_per_tma = num_elems_per_tma * sizeof_bits_v<T>;

  auto smem_swizzle = get_swizzle_portion(slayout);
  auto smem_layout  = get_nonswizzle_portion(slayout);

  auto tma_gbasis = construct_tma_gbasis<T>(gtensor, slayout, cta_v_map);
  auto aux_params = make_tma_copy_aux_params<T>(gtensor, tma_gbasis, smem_swizzle);

  auto tensor_desc = make_conv2d_tensor_desc<AuxParams>(gtensor, slayout);

  using DmaCache = Xe4DmaCache<decltype(tensor_desc), decltype(aux_params)>;
  using Traits = Copy_Traits<Xe4CopyOp<CopyOp>, cute::C<num_bits_per_tma>, DmaCache>;
  using Atom   = Copy_Atom<Traits, typename GEngine::value_type>;

  Traits tma_traits{{tensor_desc, aux_params}};

  // Return the Copy_Atom
  return Atom{tma_traits};
}

template <class SLayout, class TLayout, class VLayout, class NumElemsPerTma>
CUTE_HOST_RTC auto
make_tv_layout(SLayout const& slayout, TLayout const& cta_t_map, VLayout const& cta_v_map, NumElemsPerTma const& num_elems_per_tma)
{
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

  return layout_TV;
}

template <class CopyOp, class AuxParams, class TmaInternalType, class GTensor, class SLayout, class TLayout, class VLayout>
CUTE_HOST_RTC auto
make_xe4_copy_tiled(GTensor const& gtensor, SLayout const& slayout,
                    TLayout const& cta_t_map, VLayout const& cta_v_map)
{
  auto cta_tiler = product_each(shape(cta_v_map));
  auto atom = make_copy_atom<CopyOp, AuxParams, TmaInternalType>(gtensor, slayout, size(cta_t_map), cta_v_map);

  auto num_elems_per_tma = size<1>(typename decltype(atom)::RefLayout{}) / static_value<sizeof_bits<typename GTensor::value_type>>();
  auto layout_TV = make_tv_layout(slayout, cta_t_map, cta_v_map, num_elems_per_tma);

  return TiledCopy<decltype(atom), decltype(layout_TV), decltype(cta_tiler)>{atom};
}

template <class CopyOp, class AuxParams, class TmaInternalType, class GTensor, class SLayout, class TLayout, class VLayout>
CUTE_HOST_RTC auto
make_xe4_copy_tiled_conv2d(GTensor const& gtensor, SLayout const& slayout,
                    TLayout const& cta_t_map, VLayout const& cta_v_map)
{
  auto cta_tiler = product_each(shape(cta_v_map));
  auto atom = make_copy_atom_conv2d<CopyOp,AuxParams>(gtensor, slayout, cosize(cta_t_map), cta_v_map);

  constexpr auto num_elems_per_tma = size<1>(typename decltype(atom)::RefLayout{}) / static_value<sizeof_bits<typename GTensor::value_type>>();
  auto layout_TV = make_tv_layout(slayout, cta_t_map, cta_v_map, num_elems_per_tma);

  return TiledCopy<decltype(atom), decltype(layout_TV), decltype(cta_tiler)>{atom};
}

template <class CopyOp, class AuxParams, class TmaInternalType = void, class GTensor, class SLayout, class Tiler, class Cluster_Size>
CUTE_HOST_RTC auto
make_xe4_copy(GTensor const& gtensor, SLayout const& slayout, Tiler const& cta_tiler, Cluster_Size const& cluster_size)
{
  auto cta_t_tile = make_layout(cluster_size);
  auto cta_v_tile = make_identity_layout(shape(gtensor)).compose(cta_tiler);
  // Prefer TmaInternalType if specified. Fallback to GEngine::value_type
  using TmaType = conditional_t<is_same<void, TmaInternalType>::value, typename GTensor::value_type, TmaInternalType>;
  return make_xe4_copy_tiled<CopyOp, AuxParams, TmaType>(gtensor, slayout, cta_t_tile, cta_v_tile);
}

template <class CopyOp, class AuxParams, class TmaInternalType = void, class GTensor, class SLayout, class Tiler>
CUTE_HOST_RTC auto
make_xe4_copy(GTensor const& gtensor, SLayout const& slayout, Tiler const& cta_tiler)
{
  return make_xe4_copy<CopyOp, AuxParams, TmaInternalType>(gtensor, slayout, cta_tiler, _1{});
}

template <class CopyOp, class AuxParams, class TmaInternalType = void, class GTensor, class SLayout, class Tiler, class Cluster_Size>
CUTE_HOST_RTC auto
make_xe4_copy_conv2d(GTensor const& gtensor, SLayout const& slayout, Tiler const& cta_tiler, Cluster_Size const& cluster_size)
{
  auto cta_v_tile = make_identity_layout(shape(gtensor)).compose(cta_tiler);
  auto cta_t_tile = make_layout(cluster_size);
  // Prefer TmaInternalType if specified. Fallback to GEngine::value_type
  using TmaType = conditional_t<is_same<void, TmaInternalType>::value, typename GTensor::value_type, TmaInternalType>;
  return make_xe4_copy_tiled_conv2d<CopyOp, AuxParams, TmaType>(gtensor, slayout, cta_t_tile, cta_v_tile);
}

template <class CopyOp, class AuxParams, class TmaInternalType = void, class GTensor, class SLayout, class Tiler>
CUTE_HOST_RTC auto
make_xe4_copy_conv2d(GTensor const& gtensor, SLayout const& slayout, Tiler const& cta_tiler)
{
  return make_xe4_copy_conv2d<CopyOp, AuxParams, TmaInternalType>(gtensor, slayout, cta_tiler, _1{});
}

} // namespace detail

} // namespace cute
