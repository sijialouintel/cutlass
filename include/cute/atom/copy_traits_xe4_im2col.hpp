#pragma once

#include "cute/arch/copy_xe4_dma.hpp"
#include "cutlass/detail/layout.hpp"
#include "cute/atom/copy_traits_sm90_tma.hpp"
#include "cute/atom/copy_traits_xe4_dma.hpp"

namespace cute
{
// Utility for unpacking XE4_LOAD/STORE_IM2COL arguments into a CopyOp
template <class CopyOp>
struct XE4_IM2COL_COPY_Unpack
{
  template <class... Args,
            class TS, class SLayout,
            class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits<CopyOp, Args...> const& traits,
              Tensor<TS,SLayout>           const& src, // tile of the transformed global activation (A) tensor
              Tensor<TD,DLayout>                & dst) // shared memory tile
  {
    constexpr auto isLoadOperation = !cute::is_base_of<xe4::ASYNC_ROW_STORE_IM2COL, CopyOp>::value;

    if constexpr (isLoadOperation) {
      auto src_coord_offset = src(Int<0>{});
      auto src_coord_cwhdn_offset_srt = flatten(src_coord_offset);

      auto dst_ptr = dst.data();
      return cute::detail::explode_tuple(cute::detail::CallCOPY<CopyOp>{},
                                    traits.opargs_, tuple_seq<decltype(traits.opargs_)>{},
                                    make_tuple(dst_ptr), seq<0>{},
                                    src_coord_cwhdn_offset_srt, tuple_seq<decltype(src_coord_cwhdn_offset_srt)>{});
    } else {
      auto src_ptr = src.data();
      auto dst_coord = flatten(take<0,3>(dst(Int<0>{})));
      return cute::detail::explode_tuple(cute::detail::CallCOPY<CopyOp>{},
                                  traits.opargs_, tuple_seq<decltype(traits.opargs_)>{},
                                  make_tuple(src_ptr), seq<0>{},
                                  dst_coord, tuple_seq<decltype(dst_coord)>{});
    }
  }
};


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////// XE4_LOAD_IM2COL / XE4_STORE_IM2COL ///////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <class TensorDesc, class CoordTensor>
struct Xe4Im2ColCache {
  template <typename CopyOp>
  using OpUnpack = XE4_IM2COL_COPY_Unpack<CopyOp>;

  CUTE_HOST_DEVICE constexpr
  auto get_tensor_desc() const {
    return &tensor_desc_;
  }

  template <class GShape>
  CUTE_HOST_DEVICE constexpr
  auto get_tma_tensor([[maybe_unused]] GShape const& g_shape) const {
    return coord_tensor_;
  }

  template <typename... Args>
  CUTE_HOST_DEVICE constexpr
  auto make_args_tuple(Args&&... args) const {
    return make_tuple(&tensor_desc_, static_cast<Args&&>(args)...);
  }

  TensorDesc tensor_desc_;
  CoordTensor coord_tensor_;
};

namespace detail {
// create DMA im2col descriptor
template <class EngineA, class LayoutA, class TMALayout,
          class LowerCornerStride,
          class UpperCornerStride,
          class LowerPaddingStride,
          class UpperPaddingStride,
          class TraversalStride,
          class LowerSRTStride,
          class DilationStride>
CUTE_HOST
auto
make_im2col_tma_copy_desc(
    Tensor<EngineA, LayoutA>    const& tensor_cwhdn,       // (C,W,H,D,N)
    uint32_t                           range_c,            // TILE_C
    uint32_t                           range_whdn,         // TILE_WHDN
    TMALayout                   const& tma_layout_vt,      // TMA layout
    LowerCornerStride           const& lower_corner_whd,   // WHD offset of the "base pointer"
    UpperCornerStride           const& upper_corner_whd,   // WHD upper corner
    LowerPaddingStride          const& lower_padding_whd,  // WHD lower padding
    UpperPaddingStride          const& upper_padding_whd,  // WHD upper padding
    TraversalStride             const& stride_whd,         // WHD traversal stride
    LowerSRTStride              const& lower_srt,          // SRT offset of the "base pointer"
    DilationStride              const& stride_srt)          // SRT stride - dilation
{
  //static_assert(is_gmem<EngineA>::value, "Tensor must point to GPU global memory.");
  using value_type = typename EngineA::value_type;

  constexpr uint32_t num_total_modes   = LayoutA::rank;
  constexpr int      num_spatial_modes = num_total_modes - 2;

  // Gmem starting address
  void* gmem_address = (void*) raw_pointer_cast(tensor_cwhdn.data());

  // Gmem extents are just the tensor shape
  cute::array<uint64_t, 5> gmem_prob_shape = {1,1,1,1,1};
  for_each(make_seq<num_total_modes>{}, [&](auto i) {
    gmem_prob_shape[i] = static_cast<uint64_t>(shape<i>(tensor_cwhdn));
  });

  // Gmem strides are byte strides of the activation tensor in CWHDN order
  cute::array<uint64_t, 5> gmem_prob_stride = {0,0,0,0,0};
  for_each(make_seq<num_total_modes>{}, [&](auto i) {
    gmem_prob_stride[i] = sizeof(value_type) * stride<i>(tensor_cwhdn);
  });

  // Traversal strides are a function of the dilation shape
  // corresponding to spatial (WHD) modes.
  cute::array<uint32_t, 5> tma_traversal_strides = {1,1,1,1,1};
  for_each(make_seq<num_spatial_modes>{}, [&](auto i) {
    tma_traversal_strides[i+1] = static_cast<uint32_t>(get<i>(stride_whd));
  });

  cute::array<int32_t, num_spatial_modes> tma_lower_corner{};
  for_each(make_seq<num_spatial_modes>{}, [&](auto i) {
    tma_lower_corner[i] = static_cast<int32_t>(get<i>(lower_corner_whd));
  });

  cute::array<int32_t, num_spatial_modes> tma_upper_corner{};
  for_each(make_seq<num_spatial_modes>{}, [&](auto i) {
    tma_upper_corner[i] = static_cast<int32_t>(get<i>(upper_corner_whd));
  });


  //
  // Calculate gemm shapes and linearized shapes based on tma layout tiling.
  //

  // Compute [w, h, d, n]
  // q/p/z = (w/h/d + (upper_corner_whd - lower_corner_whd - 1)) / stride_whd + 1
  auto gemm_mn_ = cute::transform(cute::make_seq<num_spatial_modes>{}, [&](auto i) {
    return (shape<i+1>(tensor_cwhdn) + get<i>(upper_corner_whd) - get<i>(lower_corner_whd) - Int<1>{}) / get<i>(stride_whd) + Int<1>{};
  });
  auto gemm_mn = append(gemm_mn_, shape<num_spatial_modes+1>(tensor_cwhdn));

  // Compute [c, s, r, t]
  // fprop/wgrad, s/r/t = 1 + (upper_padding_whd - upper_corner_whd) / stride_srt
  // wgrad,       s/r/t = 1 + (lower_padding_whd - lower_corner_whd) / stride_srt
  auto gemm_k_ = cute::transform(cute::make_seq<num_spatial_modes>{}, [&](auto i) {
    auto padding_size = conditional_return(get<i>(stride_srt) > Int<0>{},
                                           get<i>(upper_padding_whd) - get<i>(upper_corner_whd),
                                           get<i>(lower_corner_whd)  - get<i>(lower_padding_whd));
    return Int<1>{} + padding_size / get<i>(stride_srt);
  });
  auto gemm_k = prepend(gemm_k_, shape<0>(tensor_cwhdn));

  // For fprop/dgrad kernel, gemm_shapes is ((q, p, z, n), (c, s, r, t))
  // For wgrad kernel, gemm_shapes is ((c, s, r, t), (q, p, z, n))
  auto gemm_shapes_common = make_shape(gemm_mn, gemm_k);
  /*
  auto gemm_shapes_common = make_shape(
      transform_leaf(gemm_mn, [](auto s) {

        print(cutlass::FastDivmod(s)); print("\n");
        return conditional_return(cute::is_static<decltype(s)>{}, s, cutlass::FastDivmod(s));
      }),
      gemm_k);
  */
  auto gemm_shapes = make_shape(
      basis_get(stride<0,1>(tma_layout_vt), gemm_shapes_common),
      basis_get(stride<0,0>(tma_layout_vt), gemm_shapes_common));

  // For fprop/dgrad kernel, linearized shapes is (whdn, (c, s, r, t))
  // For wgrad kernel linearized shapes is ((c, s, r, t), whdn)
  auto linear_shapes_common = make_shape(size(gemm_mn), gemm_k);
  auto linear_shapes = make_shape(
      basis_get(stride<0,1>(tma_layout_vt), linear_shapes_common),
      basis_get(stride<0,0>(tma_layout_vt), linear_shapes_common));

  //
  // Calculate gmem basis stride based on tma layout tiling.
  //

  auto tma_basis_scale = make_shape(Int<1>{}, stride_whd, Int<1>{}, stride_srt);
  auto tma_basis = elem_scale(tma_basis_scale, make_basis_like(tma_basis_scale));

  auto gbasis_strides_common = make_stride(
      append(get<1>(tma_basis), get<2>(tma_basis)),
      prepend(get<3>(tma_basis), get<0>(tma_basis)));    // ((w,h,d,n),(c,s,r,t))
  auto gbasis_strides = make_stride(
      basis_get(stride<0,1>(tma_layout_vt), gbasis_strides_common),
      basis_get(stride<0,0>(tma_layout_vt), gbasis_strides_common));

  //
  // Create tma tensor
  //
  auto lower_corner = make_arithmetic_tuple(Int<0>{}, lower_corner_whd, Int<0>{}, lower_srt);

  auto tensor_multimode = make_tensor(ArithmeticTupleIterator(lower_corner), gemm_shapes, gbasis_strides);
  auto tensor_linear = make_identity_tensor(linear_shapes);
  auto tma_tensor = make_tensor(tensor_multimode.data(), composition(
      tensor_multimode.layout(),
      tensor_linear(Int<0>{}),
      tensor_linear.layout()));

  return tma_tensor;
}

template <class CopyOp, class CMType,
          class GEngine, class GLayout,
          class SLayout,
          class VShape, class VStride,
          class LowerCornerStride,
          class UpperCornerStride,
          class LowerPaddingStride,
          class UpperPaddingStride,
          class TraversalStride,
          class LowerSRTStride,
          class DilationStride>
CUTE_HOST_RTC
auto
make_tma_atom_im2col(Tensor<GEngine,GLayout>      const& gtensor,           // Full GMEM Tensor: ((w, h, d, n), c)
                     SLayout                      const& slayout,           // CTA Tile of SMEM, potentially swizzled
                     int32_t                      const& num_multicast,     // The number of CTAs involved in multicasting
                     Layout<VShape,VStride>       const& cta_v_map,         // V: CTA val idx -> gmem mode
                     LowerCornerStride            const& lower_corner_whd,
                     UpperCornerStride            const& upper_corner_whd,
                     LowerPaddingStride           const& lower_padding_whd,
                     UpperPaddingStride           const& upper_padding_whd,
                     TraversalStride              const& stride_whd,        // traversal stride
                     LowerSRTStride               const& lower_srt,
                     DilationStride               const& stride_srt)        // dilation
{
  //
  // TMA slayout manipulation
  //
  // Invert the smem to get the largest contiguous vector in the smem layout
  auto inv_smem_layout = right_inverse(slayout);
  // trunc_smem_idx -> trunc_smem_coord

  // Map from smem idx to a gmem mode
  auto sidx_to_gmode = coalesce(composition(cta_v_map, inv_smem_layout));
  //
  // TMA gtensor manipulation
  //

  // Generate a TupleBasis for the gtensor
  auto glayout_basis = make_identity_layout(product_each(shape(gtensor)));

  // Tile the modes of gtensor with the truncated cta_v_map o inv_smem_layout_trunc
  auto tma_layout_full = flatten(composition(glayout_basis, sidx_to_gmode));

  // Truncate any incompatibilities -- no starting in the middle of gmodes
  auto smem_rank = find_if(stride(tma_layout_full), [](auto e) {
    [[maybe_unused]] auto v = basis_value(e);
    return not is_constant<1,decltype(v)>{};
  });
  // IM2COL uses a maximum of 2 modes
  constexpr int smem_tma_rank = cute::min(int(smem_rank), 2);

  // Keep only the static-1 basis modes into gmem
  auto tma_layout_trunc = take<0,smem_tma_rank>(tma_layout_full);

  // Split according to the portion each multicast CTA will be responsible for
  auto tma_layout_vt = logical_divide(tma_layout_trunc, shape_div(size(tma_layout_trunc), num_multicast));

  auto range_c    = size<0,0>(tma_layout_vt);
  auto range_whdn = size<0,1>(tma_layout_vt);
  Tensor gtensor_cwhdn = make_tensor(gtensor.data(),
                                     flatten(make_layout(make_layout(basis_get(stride<0,0>(tma_layout_vt), gtensor.shape()),
                                                                     basis_get(stride<0,0>(tma_layout_vt), gtensor.stride())),
                                                         make_layout(basis_get(stride<0,1>(tma_layout_vt), gtensor.shape()),
                                                                     basis_get(stride<0,1>(tma_layout_vt), gtensor.stride())))));
  auto tma_tensor = make_im2col_tma_copy_desc(
      gtensor_cwhdn,
      range_c,
      range_whdn,
      tma_layout_vt,
      lower_corner_whd,
      upper_corner_whd,
      lower_padding_whd,
      upper_padding_whd,
      stride_whd,
      lower_srt,
      stride_srt);

  //
  // Construct the Copy_Traits
  //
  using T = typename GEngine::value_type;
  constexpr int num_bits_per_tma = decltype(size<0, 0>(tma_layout_trunc))::value * sizeof(T) * 8;
  constexpr int num_bytes_per_tma = decltype(size<0, 0>(tma_layout_trunc))::value * sizeof(T);

  using Im2ColDesc = xe4::Im2ColDescriptor<T, cute::C<CMType::value>, num_bytes_per_tma>;
  using Im2ColCache = Xe4Im2ColCache<Im2ColDesc, decltype(tma_tensor)>;
  using Traits = Copy_Traits<Xe4CopyOp<CopyOp>, cute::C<num_bits_per_tma>, Im2ColCache>;
  using Atom = Copy_Atom<Traits, typename GEngine::value_type>;

  Im2ColDesc desc = cute::xe4::make_async_row_copy_desc<CMType, num_bytes_per_tma>(gtensor);
  Traits tma_traits{{desc, tma_tensor}};

  // Return the Copy_Atom
  return Atom{tma_traits};
}

template <class CopyOp, class CMType,
          class Engine0, class Layout0,
          class SLayout,
          class ThrLayout,
          class ValLayout,
          class CTATiler,
          class LowerCornerStride,
          class UpperCornerStride,
          class LowerPaddingStride,
          class UpperPaddingStride,
          class TraversalStride,
          class LowerSRTStride,
          class DilationStride>
CUTE_HOST_RTC
auto
make_im2col_tma_copy(Tensor<Engine0, Layout0> const& tensor_cwhdn,
                     SLayout                  const& slayout,
                     ThrLayout                const& thrlayout,
                     ValLayout                const& vallayout,
                     CTATiler                 const& cta_tiler,
                     int32_t                  const& multicast_size,
                     LowerCornerStride        const& lower_corner_whd,
                     UpperCornerStride        const& upper_corner_whd,
                     LowerPaddingStride       const& lower_padding_whd,
                     UpperPaddingStride       const& upper_padding_whd,
                     TraversalStride          const& stride_whd,
                     LowerSRTStride           const& lower_srt,
                     DilationStride           const& stride_srt)
{
  auto cta_v_tile = make_identity_layout(product_each(shape(tensor_cwhdn))).compose(cta_tiler);
  auto tma_atom = detail::make_tma_atom_im2col<CopyOp, CMType>(tensor_cwhdn,
                                                                        slayout, multicast_size, cta_v_tile,
                                                                        lower_corner_whd, upper_corner_whd, lower_padding_whd, upper_padding_whd, stride_whd, lower_srt, stride_srt);

  return make_tiled_copy(tma_atom, thrlayout, vallayout);
}

} // namespace detail
} // namespace cute
