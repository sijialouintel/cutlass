#pragma once

#include "inline_pisa.hpp"
#include "util.hpp"

namespace cute::xe4
{

////////////////////////////////////////////////////////////////////////////////////////////////////
/// ASYNC_TENSOR_LOAD: Initiates a async tensor copy from global memory to shared memory
////////////////////////////////////////////////////////////////////////////////////////////////////

struct ASYNC_TENSOR_LOAD
{
  template<class T>
  CUTE_HOST_DEVICE static void
  copy(uint64_t const* tdesc_ptr, uint64_t const* abar_ptr, T* slm_ptr, int32_t crd0, int32_t crd1)
  {
    auto coord = sycl::vec<int32_t, 2>{crd0, crd1};
    async_tensor_load<2>(tdesc_ptr, slm_space_cast(slm_ptr), coord, abar_ptr);
  }

  template<class T>
  CUTE_HOST_DEVICE static void
  copy(uint64_t const* tdesc_ptr, uint64_t const* abar_ptr, T* slm_ptr, int32_t crd0, int32_t crd1, int32_t crd2)
  {
    (void)crd2;
    copy(tdesc_ptr, abar_ptr, slm_ptr, crd0, crd1);
  }

  template<class T>
  CUTE_HOST_DEVICE static void
  copy(uint64_t const* tdesc_ptr, uint64_t const* abar_ptr, T* slm_ptr, int32_t crd0, int32_t crd1, int32_t crd2, int32_t crd3)
  {
    // Note: the coord order doesn't align with that in cutlass
    auto coord = sycl::vec<int32_t, 4>{crd0, crd2, crd3, crd1};
    async_tensor_load<4>(tdesc_ptr, slm_space_cast(slm_ptr), coord, abar_ptr);
  }
};


////////////////////////////////////////////////////////////////////////////////////////////////////
/// ASYNC_TENSOR_STORE : Initiates a async tensor copy from shared memory to global memory
////////////////////////////////////////////////////////////////////////////////////////////////////

struct ASYNC_TENSOR_STORE
{
  template<class T>
  CUTE_HOST_DEVICE static void
  copy(uint64_t const* tdesc_ptr, uint64_t const* abar_ptr, T* slm_ptr, int32_t crd0, int32_t crd1)
  {
    auto coord = sycl::vec<int32_t, 2>{crd0, crd1};
    async_tensor_store<2>(tdesc_ptr, slm_space_cast(slm_ptr), coord, abar_ptr);
  }

  template<class T>
  CUTE_HOST_DEVICE static void
  copy(uint64_t const* tdesc_ptr, uint64_t const* abar_ptr, T* slm_ptr, int32_t crd0, int32_t crd1, int32_t crd2)
  {
    (void)crd2;
    copy(tdesc_ptr, abar_ptr, slm_ptr, crd0, crd1);
  }
};


////////////////////////////////////////////////////////////////////////////////////////////////////
/// ASYNC_TENSOR_LOAD_MULTICAST: Initiates a async tensor copy from global memory to shared memory
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, typename CMType, int NumBytesPerCopy, int NumDims>
struct Im2ColDescriptor {
  T* gmem_address;
  uint32_t gmem_shape[NumDims];
  uint64_t gmem_stride[NumDims - 1];

  Im2ColDescriptor(T* address, cute::array<int, NumDims> const&shape, cute::array<int64_t, NumDims> const&stride) : gmem_address(address) {
    for (int i = 0; i < NumDims; i++)
      gmem_shape[i] = shape[NumDims - 1 - i];

    for (int i = 0; i < NumDims - 1; i++)
      gmem_stride[i] = stride[NumDims - 2 - i] * sizeof(T);
  }
};

struct ASYNC_TENSOR_LOAD_MULTICAST
{
  template<class T>
  CUTE_HOST_DEVICE static void
  copy(uint64_t const* tdesc_ptr, uint64_t const* abar_ptr, uint32_t multicast_mask, T* slm_ptr, int32_t crd0, int32_t crd1)
  {
    auto coord = sycl::vec<int32_t, 2>{crd0, crd1};
    async_tensor_load<2>(tdesc_ptr, slm_space_cast(slm_ptr), coord, abar_ptr, multicast_mask);
  }

  template<class T>
  CUTE_HOST_DEVICE static void
  copy(uint64_t const* tdesc_ptr, uint64_t const* abar_ptr, uint32_t multicast_mask, T* slm_ptr, int32_t crd0, int32_t crd1, int32_t crd2)
  {
    (void)crd2;
    copy(tdesc_ptr, abar_ptr, multicast_mask, slm_ptr, crd0, crd1);
  }
};


////////////////////////////////////////////////////////////////////////////////////////////////////
/// SLM_VLOAD: Initiates a slm load from shared memory to register
////////////////////////////////////////////////////////////////////////////////////////////////////

template<uint32_t VS>
struct SLM_VLOAD
{
  template<typename SlmType, typename RegType>
  CUTE_HOST_DEVICE static void
  copy(SlmType* slm_ptr, RegType* reg_ptr)
  {
    slm_vload<VS>(reg_ptr, slm_space_cast(slm_ptr));
  }
};


////////////////////////////////////////////////////////////////////////////////////////////////////
/// SLM_VSTORE: Initiates a slm store from register to shared memory
////////////////////////////////////////////////////////////////////////////////////////////////////

template<uint32_t VS>
struct SLM_VSTORE
{
  template<typename SlmType, typename RegType>
  CUTE_HOST_DEVICE static void
  copy(RegType* reg_ptr, SlmType* slm_ptr)
  {
    slm_vstore<VS>(slm_space_cast(slm_ptr), reg_ptr);
  }
};

/// ASYNC_ROW_LOAD_IM2COL: Initiates an im2col async row copy from global memory to shared memory
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
inline uint32_t get_copy_size(const int32_t coord, const uint32_t shape, uint32_t width_2d) {
    uint32_t left_size = (shape - coord) * sizeof(T);
    uint32_t copy_size = left_size < width_2d ? left_size : width_2d;
    return copy_size;
}

struct XE4_ASYNC_ROW_LOAD_IM2COL_4D
{
  template<class TS, class TG, class CMType, int NumBytesPerCopy, int NumDims>
  CUTE_HOST_DEVICE static void
  copy(const Im2ColDescriptor<TG, CMType, NumBytesPerCopy, NumDims>* tma_desc, uint64_t const* abar_ptr,
                              TS const* slm_ptr,
                              int32_t crd_c, int32_t crd_w, int32_t crd_h, int32_t crd_n, int32_t crd_s, int32_t crd_r)
  {
    constexpr uint32_t slm_stride = NumBytesPerCopy / sizeof(TG);
    TS const* slm_inst_ptr = slm_ptr - get_lane_id() * slm_stride;
    int32_t crd1 = crd_w + crd_s;
    int32_t crd2 = crd_h + crd_r;
    bool is_coord_valid = (crd1 >= 0) && (crd1 < tma_desc->gmem_shape[1]);
    is_coord_valid = is_coord_valid && (crd2 >= 0) && (crd2 < tma_desc->gmem_shape[2]);
    is_coord_valid = is_coord_valid && (crd_n >= 0) && (crd_n < tma_desc->gmem_shape[3]);

    uint32_t offset = crd_c * sizeof(TG) + crd1 * tma_desc->gmem_stride[0] + crd2 * tma_desc->gmem_stride[1] + crd_n * tma_desc->gmem_stride[2];
    offset = is_coord_valid ? offset : 0;
    uint32_t copy_size = is_coord_valid ? get_copy_size<TG>(crd_c, tma_desc->gmem_shape[0], NumBytesPerCopy) : 0;
    async_2d_tiled_load<CMType::value, NumBytesPerCopy>(slm_inst_ptr, tma_desc->gmem_address, offset, copy_size, abar_ptr);
  }
};

struct ASYNC_ROW_LOAD_IM2COL
{
  template<class TS, class TG, class CMType, int NumBytesPerCopy, int NumDims>
  CUTE_HOST_DEVICE static void
  copy(const Im2ColDescriptor<TG, CMType, NumBytesPerCopy, NumDims>* tma_desc, uint64_t const* abar_ptr,
                              TS const* slm_ptr,
                              int32_t crd_c, int32_t crd_w, int32_t crd_h, int32_t crd_n,
                              int32_t crd_s, int32_t crd_r)
  {
    return XE4_ASYNC_ROW_LOAD_IM2COL_4D::copy<TS, TG, CMType, NumBytesPerCopy, NumDims>(tma_desc, abar_ptr, slm_ptr,
                                         crd_c, crd_w, crd_h, crd_n,
                                         crd_s, crd_r);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// ASYNC_ROW_STORE_IM2COL: Initiates an im2col async row copy from shared memory to global memory
////////////////////////////////////////////////////////////////////////////////////////////////////

struct XE4_ASYNC_ROW_STORE_IM2COL_4D
{
  template<class TS, class TG, class CMType, int NumBytesPerCopy, int NumDims>
  CUTE_HOST_DEVICE static void
  copy(const Im2ColDescriptor<TG, CMType, NumBytesPerCopy, NumDims>* tma_desc, uint64_t const* abar_ptr, TS const* slm_ptr,
                              int32_t crd_c, int32_t crd_w, int32_t crd_h, int32_t crd_n)
  {
    constexpr uint32_t slm_stride = NumBytesPerCopy / sizeof(TG);
    TS const* slm_inst_ptr = slm_ptr - get_lane_id() * slm_stride;

    bool is_coord_valid = (crd_w >= 0) && (crd_w < tma_desc->gmem_shape[1]);
    is_coord_valid = is_coord_valid && (crd_h >= 0) && (crd_h < tma_desc->gmem_shape[2]);
    is_coord_valid = is_coord_valid && (crd_n >= 0) && (crd_n < tma_desc->gmem_shape[3]);

    uint32_t offset = crd_c * sizeof(TG) + crd_w * tma_desc->gmem_stride[0] + crd_h * tma_desc->gmem_stride[1] + crd_n * tma_desc->gmem_stride[2];
    offset = is_coord_valid ? offset : 0;
    uint32_t copy_size = is_coord_valid ? get_copy_size<TG>(crd_c, tma_desc->gmem_shape[0], NumBytesPerCopy) : 0;
    async_2d_tiled_store<CMType::value, NumBytesPerCopy>(slm_inst_ptr, tma_desc->gmem_address, offset, copy_size, abar_ptr);
  }
};

struct ASYNC_ROW_STORE_IM2COL
{
  template<class TS, class TG, class CMType, int NumBytesPerCopy, int NumDims>
  CUTE_HOST_DEVICE static void
  copy(const Im2ColDescriptor<TG, CMType, NumBytesPerCopy, NumDims>* tma_desc, uint64_t const* abar_ptr,
                              TS const* slm_ptr,
                              int32_t crd_c, int32_t crd_w, int32_t crd_h, int32_t crd_n)
  {
    return XE4_ASYNC_ROW_STORE_IM2COL_4D::copy<TS, TG, CMType, NumBytesPerCopy, NumDims>(tma_desc, abar_ptr, slm_ptr,
                                                                                         crd_c, crd_w, crd_h, crd_n);
  }
};
} // namespace cute::xe4