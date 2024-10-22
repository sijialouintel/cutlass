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
    auto coord = sycl::vec<int32_t, 2>{crd1, crd0};
    async_tensor_load<2>(tdesc_ptr, slm_space_cast(slm_ptr), coord, abar_ptr);
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
/// ASYNC_TENSOR_STORE : Initiates a async tensor copy from shared memory to global memory
////////////////////////////////////////////////////////////////////////////////////////////////////

struct ASYNC_TENSOR_STORE
{
  template<class T>
  CUTE_HOST_DEVICE static void
  copy(uint64_t const* tdesc_ptr, uint64_t const* abar_ptr, T* slm_ptr, int32_t crd0, int32_t crd1)
  {
    auto coord = sycl::vec<int32_t, 2>{crd1, crd0};
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

struct ASYNC_TENSOR_LOAD_MULTICAST
{
  template<class T>
  CUTE_HOST_DEVICE static void
  copy(uint64_t const* tdesc_ptr, uint64_t const* abar_ptr, uint32_t multicast_mask, T* slm_ptr, int32_t crd0, int32_t crd1)
  {
    auto coord = sycl::vec<int32_t, 2>{crd1, crd0};
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
/// SLM_VLOAD: Initiates a slm store from register to shared memory
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

} // namespace cute::xe4