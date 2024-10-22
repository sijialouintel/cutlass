#if defined(RELU)
#include "cutlass/epilogue/thread/xe4_relu_op.hpp"
#elif defined(CONVERSION)
#include "cutlass/epilogue/thread/xe4_conversion_op.hpp"
#endif

#include "inline_pisa.hpp"

using namespace cute;
using namespace sycl;
using namespace cutlass::epilogue::thread;
using namespace cutlass::epilogue::thread::detail;

class Epilogue {};

int main()
{
  using SrcType = uint32_t;
  using DstType = uint16_t;
  queue q;
  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

  constexpr int32_t gmem_size_x = 32;
  constexpr int32_t gmem_size_y = 32;
  constexpr int32_t gmem_size = gmem_size_x * gmem_size_y;

  auto *A_d = malloc_device<SrcType>(gmem_size, q);
  auto *B_d = malloc_device<DstType>(gmem_size, q);

  std::vector<SrcType> A_h(gmem_size);
  std::vector<DstType> B_h(gmem_size);

  for (auto i = 0; i != gmem_size; i++) {
    A_h[i] = i;
    B_h[i] = 0;
  }

  q.memcpy(A_d, A_h.data(), gmem_size * sizeof(SrcType)).wait();
  q.memcpy(B_d, B_h.data(), gmem_size * sizeof(DstType)).wait();

  constexpr int32_t tile_n = gmem_size_x;
  constexpr int32_t tile_m = gmem_size_y;
  constexpr int32_t tile_size = tile_n * tile_m;

  constexpr uint32_t group_range_Y = (gmem_size_y + tile_m - 1) / tile_m;
  constexpr uint32_t group_range_X = (gmem_size_x + tile_n - 1) / tile_n;
  range<3> local_range(1, 20, 32);
  range<3> group_range(1, group_range_Y, group_range_X);
  nd_range<3> Range(group_range * local_range, local_range);

  constexpr uint32_t slm_bytes = tile_size * sizeof(SrcType) + tile_size * sizeof(DstType);

  q.parallel_for<class Epilogue>(Range, [=](nd_item<3> item) {
    auto abar_ptr_base = allocate_abar<0,1>();
    auto tdesc_ptr_src = allocate_tdesc<0>();
    auto tdesc_ptr_dst = allocate_tdesc<1>();
    auto slm_ptr = alloc_slm_buffer<uint8_t, slm_bytes>(item.get_group());
    auto slm_ptr_src = slm_ptr;
    auto slm_ptr_dst = slm_ptr + tile_size * sizeof(SrcType);
    auto abar_ptr = abar_ptr_base;

    constexpr slm_matrix_type cm_type = slm_matrix_type::type1;
    constexpr uint32_t src_cm_size_x = 32 / sizeof(SrcType);
    constexpr uint32_t dst_cm_size_x = 32 / sizeof(DstType);
    constexpr uint32_t cm_size_y = 32;
    constexpr uint32_t cm_byte_size = 1024;
    sycl::vec<uint32_t, 2> gmem_size(gmem_size_x, gmem_size_y);
    sycl::vec<uint64_t, 1> gmem_stride_src(gmem_size_x * sizeof(SrcType));
    sycl::vec<uint64_t, 1> gmem_stride_dst(gmem_size_x * sizeof(DstType));
    sycl::vec<uint32_t, 2> roi_shape(tile_n, tile_m);
    sycl::vec<uint32_t, 2> elem_stride(1, 1);

    tensor_desc_fill_global_addr(tdesc_ptr_src, A_d);
    tensor_descriptor_fill_dim_size<2>(tdesc_ptr_src, gmem_size);
    tensor_descriptor_fill_dim_stride<2>(tdesc_ptr_src, gmem_stride_src);
    tensor_descriptor_fill_traverse_stride<2>(tdesc_ptr_src, elem_stride);
    tensor_descriptor_fill_roitensor_size<2>(tdesc_ptr_src, roi_shape);
    tensor_descriptor_fill_misc<SrcType, cm_type>(tdesc_ptr_src);

    tensor_desc_fill_global_addr(tdesc_ptr_dst, B_d);
    tensor_descriptor_fill_dim_size<2>(tdesc_ptr_dst, gmem_size);
    tensor_descriptor_fill_dim_stride<2>(tdesc_ptr_dst, gmem_stride_dst);
    tensor_descriptor_fill_traverse_stride<2>(tdesc_ptr_dst, elem_stride);
    tensor_descriptor_fill_roitensor_size<2>(tdesc_ptr_dst, roi_shape);
    tensor_descriptor_fill_misc<DstType, cm_type>(tdesc_ptr_dst);

    sycl::vec<int32_t, 2> toff(0, 0);
    uint32_t total_arrive_cnt = 1;
    int32_t expect_tx_src = tile_size * sizeof(SrcType);
    int32_t expect_tx_dst = tile_size * sizeof(DstType);
    uint32_t local_id = item.get_local_linear_id();
    if (local_id == 0) {
      abarrier_init(abar_ptr, total_arrive_cnt);
    }
    item.barrier(cl::sycl::access::fence_space::local_space);
    if (local_id == 0) {
      abarrier_workgroup_arrive_expect_tx(abar_ptr, expect_tx_src);
      async_tensor_load<2>(tdesc_ptr_src, slm_ptr_src, toff, abar_ptr);
    }
    uint32_t phase_bit = 0;
    abarrier_try_wait(abar_ptr, phase_bit);
    // do some computation
    if (local_id >= 128) {
      constexpr auto tile_mn = Shape<Int<tile_m>, Int<tile_n>> {};
      auto src_tensor = cute::make_tensor((SrcType *)slm_ptr_src, CoreMatrix::retile<SrcType>(tile_mn));
      auto dst_tensor = cute::make_tensor((DstType *)slm_ptr_dst, CoreMatrix::retile<DstType>(tile_mn));

#if defined(RELU)
      using EpilogueOp = DMAPostOPReLu<DstType, SrcType, 32, 4, 16, EpilogueAccessPattern::Pattern2>;
#elif defined(CONVERSION)
      using EpilogueOp = DMAPostOPConvert<DstType, SrcType, 32, 4, 16, EpilogueAccessPattern::Pattern2>;
#endif
      EpilogueOp conversion;
      conversion(src_tensor, dst_tensor, local_id);
    }
    item.barrier(cl::sycl::access::fence_space::local_space);
    if (local_id == 0){
      async_tensor_store<2>(tdesc_ptr_dst, slm_ptr_dst, toff, abar_ptr);
      abarrier_workgroup_arrive_expect_tx(abar_ptr, expect_tx_dst);
      phase_bit = 1;
      abarrier_try_wait(abar_ptr, phase_bit);
    }
  }).wait();

  q.memcpy(B_h.data(), B_d, gmem_size * sizeof(DstType)).wait();

#if defined(RELU)
  ReLu post_op;
#elif defined(CONVERSION)
  Conversion<DstType> post_op;
#endif

  uint32_t err_cnt = 0;
  for (auto i = 0; i != gmem_size_y; i++) {
    for (auto j = 0; j != gmem_size_x; j++) {
      uint32_t idx = i * gmem_size_x + j;
      if (B_h[idx] != static_cast<DstType>(post_op(A_h[idx]))) {
        err_cnt++;
        std::cout << " B: " << (int)B_h[idx]
                  << " mismatch with A: " << (int)post_op(A_h[idx])
                  << " at idx = " << idx << std::endl;
      }
    }
  }
  int rtn = 0;
  if (err_cnt > 0) {
    std::cout << "Test Failed!" << std::endl;
    rtn = -1;
  }
  else {
    std::cout << "Test Pass!" << std::endl;
    rtn = 0;
  }
  free(A_d, q);
  free(B_d, q);

  return rtn;
}
