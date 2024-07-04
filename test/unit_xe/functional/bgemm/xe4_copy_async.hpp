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
};

} // namespace cute::xe4