#pragma once

#include <cute/config.hpp>
#include <cute/tensor.hpp>

namespace cute
{

template <class GmemTmaBasisStrides_, class TmaGmemBasis_, class TmaSwizzle_>
struct AuxTmaParams {
  using GmemStrides  = GmemTmaBasisStrides_;    // Strides for Gmem mode -> Tma coord mode, may be dynamic
  GmemStrides g_stride_;
  using TmaGmemBasis = TmaGmemBasis_;           // Layout for Tma box shape -> Gmem mode(s), always static
  static_assert(is_static<TmaGmemBasis>::value);
  using TmaSwizzle   = TmaSwizzle_;             // Tma swizzle, always Swizzle<B,M,S>
  static_assert(is_static<TmaSwizzle>::value);
};

//
// MAKE_TMA_COPY and related
//

namespace detail {

// Custom version of coalesce that greedily combines modes only up to size-256
// Look at each element and the back of the stack (in order of priority)
// back(NewLayout)  get<I>(OldLayout)
//      s0:d0           _1:d1     =>  continue
//      _1:d0           s1:d1     =>  replace_back     s1:d1
//      s0:d0           s1:s0*d0  =>  replace_back  s0*s1:d0   if s0*s1 <= 256
//      s0:d0           s1:d1     =>  append           s1:d1
//
// @pre OldShape and OldStride are flat
template <int I, class OldShape, class OldStride, class NewShape, class NewStride>
CUTE_HOST_DEVICE constexpr
auto
coalesce_256_impl(OldShape const& old_shape, OldStride const& old_stride,
                  NewShape const& new_shape, NewStride const& new_stride)
{
  if constexpr (I == rank_v<OldShape>) {
    // Base case, we're done
    if constexpr (is_constant<1, NewShape>::value) {
      return Layout<_1,_0>{};
    } else {
      return Layout<NewShape,NewStride>{new_shape,new_stride};
    }
  } else if constexpr (is_constant<1, decltype(get<I>(old_shape))>::value) {
    // shape<I>(layout) == _1, skip it and continue
    return coalesce_256_impl<I+1>(old_shape, old_stride, new_shape, new_stride);
  } else if constexpr (is_constant<1, NewShape>::value) {
    // Replace our shape-1 with anything (Can only happen on input new_shape/new_stride)
    return coalesce_256_impl<I+1>(old_shape, old_stride, get<I>(old_shape), get<I>(old_stride));
  } else if constexpr (is_constant<true, decltype(back(new_shape) * back(new_stride) == get<I>(old_stride) &&
                                                  get<I>(old_shape) * back(new_shape) <= Int<256>{})>::value) {
    // Merge modes because the shapes and strides match and the merge is 256 or less
    return coalesce_256_impl<I+1>(old_shape, old_stride,
                                  replace_back(new_shape, get<I>(old_shape) * back(new_shape)),
                                  new_stride);
  } else {
    // Can't replace or merge, so append a new mode
    return coalesce_256_impl<I+1>(old_shape, old_stride,
                                  append(new_shape,  get<I>(old_shape)),
                                  append(new_stride, get<I>(old_stride)));
  }

  CUTE_GCC_UNREACHABLE;
}

// Combine all the modes that are possible to combine
// Does not respect the profile of the layout, but does preserve total size
template <class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
auto
coalesce_256(Layout<Shape,Stride> const& layout)
{
  auto flat_shape  = flatten(layout.shape());
  auto flat_stride = flatten(layout.stride());
  return coalesce_256_impl<1>(flat_shape, flat_stride, get<0>(flat_shape), get<0>(flat_stride));
}

template <class TmaInternalType,
          class GEngine, class GLayout,
          class SShape, class SStride,
          class VShape, class VStride>
CUTE_HOST_DEVICE constexpr
auto
construct_tma_gbasis(Tensor<GEngine,GLayout> const& gtensor,       // The original GMEM Tensor
                     Layout<SShape,SStride>  const& slayout,       // The layout of SMEM
                     Layout<VShape,VStride>  const& cta_v_map)     // smem_idx to hier gmode
{
  //
  // TMA parameter checking
  //

  // CUTE_STATIC_ASSERT_V(product_each(shape(slayout)) == product_each(shape(cta_v_map)),
  //                      "TMA requires CTA_Tile and SLayout top-level shape equivalence.");
  CUTE_STATIC_ASSERT_V(size(slayout) == size(cta_v_map),
                       "TMA requires CTA_Tile and SLayout top-level size equivalence.");

#if 0
  print("gtensor         : "); print(gtensor); print("\n");
  print("slayout         : "); print(slayout); print("\n");
  print("cta_v_map       : "); print(cta_v_map); print("\n");
#endif

  //
  // TMA slayout manipulation
  //

  // Invert the smem to get the largest contiguous vector in the smem layout
  // smem idx -> smem coord
  auto inv_smem_layout = right_inverse(get_nonswizzle_portion(slayout));

  // Compose with the V-Map to convert smem coord (CTA val idx) to gmem mode
  // smem idx -> gmem mode
  auto sidx2gmode_full = coalesce(composition(cta_v_map, inv_smem_layout));

#if 0
  print("inv_smem_layout : "); print(inv_smem_layout); print("\n");
  print("sidx2gmode_full : "); print(sidx2gmode_full); print("\n");
#endif

  //
  // TMA gtensor truncation
  //

  // Truncate any incompatibilities -- no starting in the middle of gmodes
  auto smem_rank = find_if(stride(sidx2gmode_full), [](auto e) {
    [[maybe_unused]] auto v = basis_value(e);
    return not is_constant<1,decltype(v)>{};
  });
  static_assert(smem_rank > 0, "Could not find a common tile-gmem vectorization. Does the Tile select out major GMEM modes?");

  // Keep only the static-1 basis modes into gmem
  auto sidx2gmode = take<0,smem_rank>(sidx2gmode_full);

#if 0
  print("smem_rank  : "); print(smem_rank); print("\n");
  print("sidx2gmode : "); print(sidx2gmode); print("\n");
#endif

  //
  // TMA gtensor manipulation
  //

  // The smem vector is the same units as gtensor, so compose first and then recast
  // tma_val_idx:gmem_strides
  auto tile_gstride = recast<TmaInternalType>(gtensor.compose(sidx2gmode)).layout();
  // Coalesce modes up to size-256 (the maximum TMA box extent in units of TmaInternalType)
  // tma_box_shape:gmem_strides
  auto tma_gstride  = coalesce_256(tile_gstride);

  // Perform the tiling, recast, and coalesce to the gmem vector again, but with indirections to the gtensor modes
  auto gbasis = make_identity_layout(shape(gtensor));
  auto tile_gbasis_tmp = gbasis.compose(sidx2gmode);

  // Instead of the recast (gbasis doesn't have type info), replace the shape with the already-recasted shape
  // tma_box_shape:gmem_mode
  auto tile_gbasis = make_layout(shape(tile_gstride), stride(tile_gbasis_tmp));

  // "Coalesce" the tile basis into a compatible shape with the tma_gstride
  auto tma_gbasis_tile = tile_gbasis.compose(make_layout(wrap(shape(tma_gstride))));

  // Recast the original tensor for shape/stride inspections
  Tensor gtensor_T = recast<TmaInternalType>(gtensor);

  // Find missing bases that don't appear in tile_gbasis
  auto tile_gbasis_remaining_stride = filter_tuple(flatten(shape (gtensor_T)), flatten(stride(gtensor_T)),
                                                   flatten(stride(gbasis)),
                                                   [&](auto s, auto d, auto e)
  {
    if constexpr (is_constant<1, decltype(s)>::value || is_constant<0, decltype(d)>::value) {
      return cute::tuple<>{};          // If size-1 or stride-0, then don't append
    } else {
      using E = decltype(e);
      auto has_e = any_of(flatten(stride(tma_gbasis_tile)), [] (auto tb) { return tb == E{}; });
      if constexpr (decltype(has_e)::value) {
        return cute::tuple<>{};        // If d was found, then don't append
      } else {
        return cute::tuple<E>(e);      // Else, this is missing so append
      }
    }
  });

  // Append the remaining basis modes that contribute to the TMA with size-1
  auto tile_gbasis_remaining_shape = repeat<rank(tile_gbasis_remaining_stride)>(Int<1>{});
  auto tma_gbasis_full = make_layout(tuple_cat(wrap( shape(tma_gbasis_tile)), wrap(tile_gbasis_remaining_shape )),
                                     tuple_cat(wrap(stride(tma_gbasis_tile)), wrap(tile_gbasis_remaining_stride)));

  // Group the trailing modes to make this max rank-5 -- TMA rank limitation
  // tma_box_shape:gmem_mode
  auto tma_gbasis = group<cute::min(rank(tma_gbasis_full),4),-1>(tma_gbasis_full);

#if 0
  print("tile_gstride : "); print(tile_gstride); print("\n");
  print("tma_gstride  : "); print(tma_gstride); print("\n");
  print("gbasis       : "); print(gbasis); print("\n");
  print("tile_gbasis  : "); print(tma_gbasis_tile); print("\n");
  print("tma_gbasis   : "); print(tma_gbasis); print("\n");
#endif

  return tma_gbasis;
}

template <class GEngine, class GLayout,
          class TmaGmemBasisStride,
          class ShapeT, size_t TmaRank>
CUTE_HOST_DEVICE constexpr
void
fill_tma_gmem_shape_stride(Tensor<GEngine,GLayout>   const& gtensor,           // Gmem Shapes and Strides, in units of TmaInternalType
                           TmaGmemBasisStride        const& tma_gbasis_stride, // Map Tma mode idx -> Gmem mode(s)
                           cute::array<ShapeT,   TmaRank> & gmem_prob_shape,   // Tma Shapes, uint32_t or uin64_t
                           cute::array<uint64_t, TmaRank> & gmem_prob_stride)  // Tma Strides
{
  static_assert(is_tuple<TmaGmemBasisStride>::value);
  static_assert(is_same<uint32_t, ShapeT>::value || is_same<uint64_t, ShapeT>::value);

  using TmaInternalType = typename GEngine::value_type;
  constexpr int tma_rank = decltype(rank(tma_gbasis_stride))::value;
  static_assert(TmaRank >= tma_rank);

  auto gmem_shape  =  shape(gtensor);
  auto gmem_stride = stride(gtensor);
  // Use the indirections in tma_gbasis_stride into gtensor to construct the tma gmem shapes/strides
  for_each(make_seq<tma_rank>{}, [&](auto i) {
    constexpr int tma_i_rank = decltype(rank<i>(tma_gbasis_stride))::value;
    if constexpr (tma_i_rank == 1) {
      // Trivial contribution of this gmem mode to this tma mode
      auto ej = unwrap(get<i>(tma_gbasis_stride));
      gmem_prob_shape[i]  = basis_get(ej, gmem_shape);
      gmem_prob_stride[i] = basis_get(ej, gmem_stride);
    } else {
      // Apply a recurrence to each gmem mode that contributes to this tma mode
      for_each(get<i>(tma_gbasis_stride), [&](auto ej) {
        // Problem shape
        uint64_t shape_j  = basis_get(ej, gmem_shape);
        // Problem stride (in bytes)
        uint64_t stride_j = basis_get(ej, gmem_stride);
        uint64_t old_stride = gmem_prob_stride[i];
        gmem_prob_stride[i] = gcd(gmem_prob_stride[i], stride_j);

        if (gmem_prob_stride[i] != 0) {
          // Recurrence: g_shape = (s_i - 1) * (d_i / gcd_j d_j) + 1
          gmem_prob_shape[i] = (gmem_prob_shape[i]-1) * (old_stride / gmem_prob_stride[i])
                             +            (shape_j-1) * (stride_j   / gmem_prob_stride[i])
                             + 1;
        } else {
          gmem_prob_shape[i] = shape_j;
        }
      });
    }
  });
}

// Overload for an existing Copy_Traits
template <class GEngine, class GLayout,
          class Op, class Bits, class Aux,
          class ShapeT, size_t TmaRank>
CUTE_HOST_DEVICE constexpr
void
fill_tma_gmem_shape_stride(Copy_Traits<Op,Bits,Aux>  const& tma_traits,
                           Tensor<GEngine,GLayout>   const& gtensor,           // Gmem Shapes and Strides, value_type = TmaInternalType
                           cute::array<ShapeT,   TmaRank> & gmem_prob_shape,   // Tma Shapes, uint32_t or uin64_t
                           cute::array<uint64_t, TmaRank> & gmem_prob_stride)  // Tma Strides
{
  return fill_tma_gmem_shape_stride(gtensor, stride(typename Aux::TmaGmemBasis{}),
                                    gmem_prob_shape, gmem_prob_stride);
}

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

} // namespace detail
} // namespace cute
