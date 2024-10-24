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
    if constexpr (CopyOp::isLoadOperation) {
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


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////// ASYNC_TENSOR_LOAD / ASYNC_TENSOR_STORE ///////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


struct ASYNC_TENSOR_LOAD_OP : xe4::ASYNC_TENSOR_LOAD {
  static constexpr bool isLoadOperation = true;
};

struct ASYNC_TENSOR_STORE_OP : xe4::ASYNC_TENSOR_STORE {
  static constexpr bool isLoadOperation = false;
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
  with(ABarrier const* abar_ptr, [[maybe_unused]] uint32_t const& multicast_mask = 0) const {
    return {{}, {tdesc_ptr_, abar_ptr}};
  }

  template <class GShape>
  CUTE_HOST_DEVICE constexpr
  auto get_tma_tensor(GShape const& g_shape) const {
    return make_counting_tensor(make_layout(g_shape, gbasis_));
  }

  template <class GShape>
  CUTE_HOST_DEVICE constexpr
  auto get_tma_tensor_test(GShape const& g_shape) const {
    if (DEBUG_THREAD) {
      PRINT(g_shape);
      PRINT(gbasis_);
    }
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

template <class CopyOp, class NumBitsPerTMA, class TensorDesc, class ABarrier, class Unused>
struct Copy_Traits<CopyOp, NumBitsPerTMA, TensorDesc, ABarrier, Unused> : XE4_COPY_Unpack<CopyOp>
{
  using ThrID     = Layout<_1>;
  using SrcLayout = Layout<Shape<_1, NumBitsPerTMA>>;
  using DstLayout = SrcLayout;
  using RefLayout = SrcLayout;

  tuple<TensorDesc const*, ABarrier const*> const opargs_;
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////// ASYNC_TENSOR_LOAD_MULTICAST / ASYNC_TENSOR_STORE_MULTICAST /////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct ASYNC_TENSOR_LOAD_MULTICAST_OP : xe4::ASYNC_TENSOR_LOAD_MULTICAST  {
  static constexpr bool isLoadOperation = true;
};

template <class NumBitsPerTMA, class TensorDesc, class GBasis>
struct Copy_Traits<xe4::ASYNC_TENSOR_LOAD_MULTICAST, NumBitsPerTMA, TensorDesc, GBasis>
{
  using ThrID     = Layout<_1>;
  using SrcLayout = Layout<Shape<_1, NumBitsPerTMA>>;
  using DstLayout = SrcLayout;
  using RefLayout = SrcLayout;

  TensorDesc* tdesc_ptr_ = nullptr;
  GBasis gbasis_;

  template<class ABarrier>
  CUTE_HOST_DEVICE constexpr
  Copy_Traits<ASYNC_TENSOR_LOAD_MULTICAST_OP, NumBitsPerTMA, TensorDesc, ABarrier, bool>
  with(ABarrier const* abar_ptr, uint32_t const& multicast_mask) const {
    return {{}, {tdesc_ptr_, abar_ptr, multicast_mask}};
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

template <class NumBitsPerTMA, class TensorDesc, class ABarrier, class Unused>
struct Copy_Traits<ASYNC_TENSOR_LOAD_MULTICAST_OP, NumBitsPerTMA, TensorDesc, ABarrier, Unused> : XE4_COPY_Unpack<ASYNC_TENSOR_LOAD_MULTICAST_OP>
{
  using ThrID     = Layout<_1>;
  using SrcLayout = Layout<Shape<_1, NumBitsPerTMA>>;
  using DstLayout = SrcLayout;
  using RefLayout = SrcLayout;

  tuple<TensorDesc const*, ABarrier const*, uint32_t> const opargs_;
};

template<class CopyOp, class NumBitsPerTMA, class TensorDesc, class GBasis>
auto make_copy_traits(TensorDesc* tensor_desc, GBasis gbasis)
{
  return Copy_Traits<CopyOp, NumBitsPerTMA, TensorDesc, GBasis>{tensor_desc, gbasis};
}

namespace detail {

template<slm_matrix_type cmType_, class tdescPtr_, int tdescIdx_>
struct AuxParams {
  using tdescPtr = tdescPtr_;
  static constexpr int tdescIdx = tdescIdx_;
  static constexpr slm_matrix_type cmType = cmType_;
};

template <class GmemTmaBasisStrides_, class TmaGmemBasis_>
struct AuxConvParams {
  using GmemStrides  = GmemTmaBasisStrides_;    // Strides for Gmem mode -> Tma coord mode, may be dynamic
  GmemStrides g_stride_;
  using TmaGmemBasis = TmaGmemBasis_;           // Layout for Tma box shape -> Gmem mode(s), always static
  static_assert(is_static<TmaGmemBasis>::value);
};

template <class AuxParams, bool isTransposed, class GTensor, class SLayout>
CUTE_HOST_DEVICE auto
make_tensor_desc(GTensor const& gtensor, SLayout const& slayout, uint32_t coop_size)
{
  using T = typename GTensor::value_type;

  constexpr int dim_x = isTransposed ? 0 : 1;
  constexpr int dim_y = dim_x ^ 1;

  uint32_t width = size<dim_x>(gtensor);
  uint32_t height = size<dim_y>(gtensor);
  uint32_t block_width = size<dim_x>(slayout);
  uint32_t block_height = size<dim_y>(slayout) / coop_size;

  auto tdesc_ptr = allocate_tdesc<AuxParams::tdescIdx, typename AuxParams::tdescPtr>();
  tensor_desc_fill_global_addr(tdesc_ptr, gtensor.data());
  tensor_descriptor_fill_dim_size<2>(tdesc_ptr, {width, height});
  tensor_descriptor_fill_dim_stride<2>(tdesc_ptr, width * sizeof(T));
  tensor_descriptor_fill_traverse_stride<2>(tdesc_ptr, sycl::vec<uint32_t, 2>{1, 1});
  tensor_descriptor_fill_roitensor_size<2>(tdesc_ptr, {block_width, block_height});
  tensor_descriptor_fill_misc<T, AuxParams::cmType>(tdesc_ptr);

  return tdesc_ptr;
}

template <class AuxParams, class GTensor, class SLayout>
CUTE_HOST_DEVICE auto
make_conv2d_tensor_desc(GTensor const& gtensor, SLayout const& slayout, uint32_t coop_size = 1)
{
  using T = typename GTensor::value_type;
  
  sycl::vec<uint32_t, 4> gmem_shape {(uint32_t)shape<1,0>(gtensor), 
    (uint32_t)shape<1,1>(gtensor), (uint32_t)shape<1,2>(gtensor), (uint32_t)shape<0>(gtensor)};
  sycl::vec<uint64_t, 3> gmem_stride {stride<1,1>(gtensor) * sizeof(T), stride<1,2>(gtensor) * sizeof(T),
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

// Use a sidx2gmode to read through the GMEM tensor
//   and construct a TMA Descriptor for the resulting instruction
// At the same time, construct the Tma Tensor's Stride to generate
//   the TMA coordinates that the instruction consumes.
//
template <class TmaInternalType,
          class GEngine, class GLayout,
          class TShape, class TStride>
CUTE_HOST_RTC
auto
make_conv_copy_aux_params(Tensor<GEngine,GLayout> const& gtensor,         // The original GMEM Tensor
                   Layout<TShape,TStride>  const& tma_gbasis,      // TMA mode -> GMEM mode mapping
                   uint32_t                       num_multicast)   // The number of CTAs in multicasting
{
  //
  // TMA desc creation
  //

  constexpr int tma_dim = decltype(rank(tma_gbasis))::value;

  //
  // TMA gmem desc info
  //

  // Recast the original tensor for shape/stride inspections
  Tensor gtensor_T = recast<TmaInternalType>(gtensor);

  void* gmem_address = (void*) raw_pointer_cast(gtensor_T.data());
  auto  gmem_layout  = gtensor_T.layout();

  cute::array<uint64_t, 5> gmem_prob_shape  = {1,1,1,1,1};
  cute::array<uint64_t, 5> gmem_prob_stride = {0,0,0,0,0};

  fill_tma_gmem_shape_stride(gtensor_T, stride(tma_gbasis), gmem_prob_shape, gmem_prob_stride);

  assert((reinterpret_cast<uint64_t>(gmem_address) & 0b1111) == 0);  // Address must be 16B-aligned

  assert(gmem_prob_shape[0] >= (uint64_t(1)));               // Size must be min 1
  assert(gmem_prob_shape[0] <= (uint64_t(1) << 32));         // Size must be max 2^32
  assert(gmem_prob_shape[1] >= (uint64_t(1)));               // Size must be min 1
  assert(gmem_prob_shape[1] <= (uint64_t(1) << 32));         // Size must be max 2^32
  assert(gmem_prob_shape[2] >= (uint64_t(1)));               // Size must be min 1
  assert(gmem_prob_shape[2] <= (uint64_t(1) << 32));         // Size must be max 2^32
  assert(gmem_prob_shape[3] >= (uint64_t(1)));               // Size must be min 1
  assert(gmem_prob_shape[3] <= (uint64_t(1) << 32));         // Size must be max 2^32
  assert(gmem_prob_shape[4] >= (uint64_t(1)));               // Size must be min 1
  assert(gmem_prob_shape[4] <= (uint64_t(1) << 32));         // Size must be max 2^32

  // TMA descriptor does not store the zeroth stride and assumes it is 1 (TmaInternalType element).
  assert(gmem_prob_stride[0] == 1 && "Majorness of smem doesn't match majorness of gmem");

  // convert strides to byte strides
  for(uint64_t& stride : gmem_prob_stride) {
    stride = (stride * sizeof_bits_v<TmaInternalType>) / 8;
  }

  // Assert the byte strides. Tma Descriptor uses byte strides
  assert((gmem_prob_stride[1]) < (uint64_t(1) << 40));       // Stride must be max 2^40
  assert((gmem_prob_stride[1] & 0b1111) == 0);               // Stride must be multiple of 16B (128b)
  assert((gmem_prob_stride[2]) < (uint64_t(1) << 40));       // Stride must be max 2^40
  assert((gmem_prob_stride[2] & 0b1111) == 0);               // Stride must be multiple of 16B (128b)
  assert((gmem_prob_stride[3]) < (uint64_t(1) << 40));       // Stride must be max 2^40
  assert((gmem_prob_stride[3] & 0b1111) == 0);               // Stride must be multiple of 16B (128b)
  assert((gmem_prob_stride[4]) < (uint64_t(1) << 40));       // Stride must be max 2^40
  assert((gmem_prob_stride[4] & 0b1111) == 0);               // Stride must be multiple of 16B (128b)

  //
  // TMA smem desc info
  //

  cute::array<uint32_t, 5> smem_box_shape  = {1,1,1,1,1};
  cute::array<uint32_t, 5> smem_box_stride = {1,1,1,1,1};
  // The smem box is simply given by the sizes of the modes in tma_gbasis
  for_each(make_seq<tma_dim>{}, [&](auto i) {
    smem_box_shape[i] *= size<i>(tma_gbasis);
  });
  // Finally, truncate the tma box by the num_multicast
  for (uint32_t i = tma_dim-1, multicast = num_multicast; multicast > 1; --i) {
    assert(smem_box_shape[i] % multicast == 0 || multicast % smem_box_shape[i] == 0);
    uint32_t new_mult = ceil_div(multicast, smem_box_shape[i]);
    smem_box_shape[i] = ceil_div(smem_box_shape[i], multicast);
    multicast = new_mult;
  }

  assert(smem_box_shape[0] >= (uint32_t(1)));                // Size must be min 1
  assert(smem_box_shape[0] <= (uint32_t(1) << 8));           // Size must be max 2^8 = 256
  assert(smem_box_shape[1] >= (uint32_t(1)));                // Size must be min 1
  assert(smem_box_shape[1] <= (uint32_t(1) << 8));           // Size must be max 2^8 = 256
  assert(smem_box_shape[2] >= (uint32_t(1)));                // Size must be min 1
  assert(smem_box_shape[2] <= (uint32_t(1) << 8));           // Size must be max 2^8 = 256
  assert(smem_box_shape[3] >= (uint32_t(1)));                // Size must be min 1
  assert(smem_box_shape[3] <= (uint32_t(1) << 8));           // Size must be max 2^8 = 256
  assert(smem_box_shape[4] >= (uint32_t(1)));                // Size must be min 1
  assert(smem_box_shape[4] <= (uint32_t(1) << 8));           // Size must be max 2^8 = 256

  assert(smem_box_stride[0] >= (uint32_t(1)));               // Stride must be min 1
  assert(smem_box_stride[0] <= (uint32_t(8)));               // Stride must be max 2^3 = 8
  assert(smem_box_stride[1] >= (uint32_t(1)));               // Stride must be min 1
  assert(smem_box_stride[1] <= (uint32_t(8)));               // Stride must be max 2^3 = 8
  assert(smem_box_stride[2] >= (uint32_t(1)));               // Stride must be min 1
  assert(smem_box_stride[2] <= (uint32_t(8)));               // Stride must be max 2^3 = 8
  assert(smem_box_stride[3] >= (uint32_t(1)));               // Stride must be min 1
  assert(smem_box_stride[3] <= (uint32_t(8)));               // Stride must be max 2^3 = 8
  assert(smem_box_stride[4] >= (uint32_t(1)));               // Stride must be min 1
  assert(smem_box_stride[4] <= (uint32_t(8)));               // Stride must be max 2^3 = 8

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

  if(DEBUG_THREAD) {
      PRINT(gmem_tma_basis_stride);
  }
    


  using AuxParams = AuxConvParams<decltype(gmem_tma_basis_stride),
                                 decltype(tma_gbasis)>;
  return AuxParams{gmem_tma_basis_stride};
}

template <class CopyOp, class AuxParams, class GEngine, class GLayout, class SLayout>
CUTE_HOST_DEVICE auto
make_copy_atom(Tensor<GEngine, GLayout> const& gtensor, SLayout const& slayout, uint32_t coop_size)
{
  using T = typename GEngine::value_type;

  auto num_elems_per_tma = size<0>(group<0, 2>(slayout));
  constexpr uint32_t num_bits_per_tma = num_elems_per_tma * sizeof_bits_v<T>;

  constexpr bool isTransposed = is_mn_major(GLayout{});
  auto gbasis = get_gbasis<isTransposed>();
  auto tensor_desc= make_tensor_desc<AuxParams, isTransposed>(gtensor, slayout, coop_size);
  auto copy_traits = make_copy_traits<CopyOp, C<num_bits_per_tma>>(tensor_desc, gbasis);
  return Copy_Atom<decltype(copy_traits), T>{copy_traits};
}

template <class CopyOp, class AuxParams, class GEngine, class GLayout, class SLayout, class VLayout>
CUTE_HOST_DEVICE auto
make_copy_atom_conv2d(Tensor<GEngine, GLayout> const& gtensor, SLayout const& slayout,
uint32_t coop_size, VLayout const& cta_v_map)
{
  using T = typename GEngine::value_type;

  auto num_elems_per_tma = size<0>(group<0, 2>(slayout));
  constexpr uint32_t num_bits_per_tma = num_elems_per_tma * sizeof_bits_v<T>;

  auto tma_gbasis = construct_tma_gbasis<T>(gtensor, slayout, cta_v_map);
  auto aux_params = make_conv_copy_aux_params<T>(gtensor, tma_gbasis, coop_size);
  if (thread0()) {
    PRINT(aux_params.g_stride_);
  }
  auto tensor_desc = make_conv2d_tensor_desc<AuxParams>(gtensor, slayout);
  auto copy_traits = make_copy_traits<CopyOp, C<num_bits_per_tma>>(tensor_desc, aux_params.g_stride_);
  return Copy_Atom<decltype(copy_traits), T>{copy_traits};
}

template <class CopyOp, class AuxParams, class GTensor, class SLayout, class TLayout, class VLayout>
CUTE_HOST_RTC auto
make_xe4_copy_tiled(GTensor const& gtensor, SLayout const& slayout,
                    TLayout const& cta_t_map, VLayout const& cta_v_map)
{
  auto cta_tiler = product_each(shape(cta_v_map));
  auto atom = make_copy_atom<CopyOp,AuxParams>(gtensor, slayout, size(cta_t_map));

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

template <class CopyOp, class AuxParams, class GTensor, class SLayout, class TLayout, class VLayout>
CUTE_HOST_RTC auto
make_xe4_copy_tiled_conv2d(GTensor const& gtensor, SLayout const& slayout,
                    TLayout const& cta_t_map, VLayout const& cta_v_map)
{
  auto atom = make_copy_atom_conv2d<CopyOp,AuxParams>(gtensor, slayout, cosize(cta_t_map), cta_v_map);

  auto cta_tiler = product_each(shape(cta_v_map));

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

  if (DEBUG_THREAD) {
    print("gtensor : "); print(gtensor); print("\n");
    print("slayout : "); print(slayout); print("\n");
    print("cta_t_map : "); print(cta_t_map); print("\n");
    print("cta_v_map : "); print(cta_v_map); print("\n");
    print("cta_tiler : "); print(cta_tiler); print("\n");
    print("layout_v : "); print(layout_v); print("\n");
    print("layout_V : "); print(layout_V); print("\n");
    print("layout_t : "); print(layout_t); print("\n");
    print("layout_T : "); print(layout_T); print("\n");
    print("layout_TV : "); print(layout_TV); print("\n");
  }
  

  return TiledCopy<decltype(atom), decltype(layout_TV), decltype(cta_tiler)>{atom};
}

template <class CopyOp, class AuxParams, class GTensor, class SLayout, class Tiler, class Cluster_Size>
CUTE_HOST_RTC auto
make_xe4_copy(GTensor const& gtensor, SLayout const& slayout, Tiler const& cta_tiler, Cluster_Size const& cluster_size)
{
  auto cta_t_tile = make_layout(cluster_size);
  auto cta_v_tile = make_identity_layout(shape(gtensor)).compose(cta_tiler);

  return make_xe4_copy_tiled<CopyOp,AuxParams>(gtensor, slayout, cta_t_tile, cta_v_tile);
}

template <class CopyOp, class AuxParams, class GTensor, class SLayout, class Tiler>
CUTE_HOST_RTC auto
make_xe4_copy(GTensor const& gtensor, SLayout const& slayout, Tiler const& cta_tiler)
{
  return make_xe4_copy<CopyOp,AuxParams>(gtensor, slayout, cta_tiler, _1{});
}

template <class CopyOp, class AuxParams, class GTensor, class SLayout, class Tiler, class Cluster_Size>
CUTE_HOST_RTC auto
make_xe4_copy_conv2d(GTensor const& gtensor, SLayout const& slayout, Tiler const& cta_tiler, Cluster_Size const& cluster_size)
{
  auto cta_v_tile = make_identity_layout(shape(gtensor)).compose(cta_tiler);
  auto cta_t_tile = make_layout(cluster_size);

  return make_xe4_copy_tiled_conv2d<CopyOp,AuxParams>(gtensor, slayout, cta_t_tile, cta_v_tile);
}

template <class CopyOp, class AuxParams, class GTensor, class SLayout, class Tiler>
CUTE_HOST_RTC auto
make_xe4_copy_conv2d(GTensor const& gtensor, SLayout const& slayout, Tiler const& cta_tiler)
{
  return make_xe4_copy_conv2d<CopyOp,AuxParams>(gtensor, slayout, cta_tiler, _1{});
}

} // namespace detail

} // namespace cute
