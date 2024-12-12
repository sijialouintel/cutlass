#include <sycl/sycl.hpp>
#include <vector>
#include <random>
#include <iostream>
#include <algorithm>

#include "cute/tensor.hpp"
#include "cute/arch/mma_xe4.hpp"
#include "cutlass/layout/matrix.h"
#include "cutlass/detail/layout.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#include "validation.hpp"

using namespace cute;
using namespace sycl;
using namespace cute::xe4;
using namespace cutlass::epilogue;
using namespace cutlass::gemm::collective;
using namespace cutlass::epilogue::collective;
using namespace cutlass::epilogue::collective::detail;
using namespace cutlass::epilogue::thread;

struct GEMM_BF16_BF8_FP4 {
  using dtypeA = bf8;
  using dtypeB = fp4_e3m0x2;
  using dtypeAcc = float;
  using dtypeC = bf16;
  static constexpr mem_layout layout_a = mem_layout::row_major;
  static constexpr mem_layout layout_b = mem_layout::col_major;
  static constexpr uint32_t wg_m = 256;
  static constexpr uint32_t wg_n = 512;
  static constexpr uint32_t wg_k = 256;
};

struct GEMM_BF16_FP4_FP4 {
  using dtypeA = fp4_e3m0x2;
  using dtypeB = fp4_e3m0x2;
  using dtypeAcc = float;
  using dtypeC = bf16;
  static constexpr mem_layout layout_a = mem_layout::row_major;
  static constexpr mem_layout layout_b = mem_layout::col_major;
  static constexpr uint32_t wg_m = 256;
  static constexpr uint32_t wg_n = 512;
  static constexpr uint32_t wg_k = 512;
};

struct GEMM_BF16_BF8_BF8 {
  using dtypeA = bf8;
  using dtypeB = bf8;
  using dtypeAcc = float;
  using dtypeC = bf16;
  static constexpr mem_layout layout_a = mem_layout::row_major;
  static constexpr mem_layout layout_b = mem_layout::col_major;
  static constexpr uint32_t wg_m = 256;
  static constexpr uint32_t wg_n = 512;
  static constexpr uint32_t wg_k = 256;
};

template <typename test>
void run_test() {
  queue q;
  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

  int mat_m = 512;
  int mat_n = 1024;
  int mat_k = 2048;
  int mat_l = 1;
  constexpr uint32_t wg_m = test::wg_m;
  constexpr uint32_t wg_n = test::wg_n;
  constexpr uint32_t wg_k = test::wg_k;

  constexpr uint32_t stage = 3;
  constexpr uint32_t split_b = 2;
  static_assert(wg_n % split_b == 0);
  constexpr bool a_blockscaling = true;
  constexpr bool b_blockscaling = true;

  using dtypeA = typename test::dtypeA;
  using dtypeB = typename test::dtypeB;
  using dtypeAcc = typename test::dtypeAcc;
  using dtypeC = typename test::dtypeC;
  constexpr mem_layout layout_a = test::layout_a;
  constexpr mem_layout layout_b = test::layout_b;

  constexpr uint32_t mx_scale_elem_num = 32;
  const int meta_k = std::max(32u, ceil_div(mat_k, mx_scale_elem_num));
  constexpr uint32_t wg_meta_k_step = wg_k / mx_scale_elem_num;
  constexpr uint32_t wg_meta_k = (wg_meta_k_step + 31) / 32 * 32;

  constexpr uint32_t num_packed_a = packed_num<dtypeA>::value;
  constexpr uint32_t num_packed_b = packed_num<dtypeB>::value;

  uint32_t sizeA = mat_m * (mat_k / num_packed_a);
  uint32_t sizeB = mat_n * (mat_k / num_packed_b);
  uint32_t sizeC = mat_m * mat_n;
  uint32_t sizeMetaA = mat_m * meta_k;
  uint32_t sizeMetaB = mat_n * meta_k;

  using dtypeMeta = e8m0;
  using mat_desc_t = uint32_t;
  using meta_desc_t = uint64_t;
  using abar_ptr_t = uint64_t*;
  using tdesc_ptr_t = uint64_t*;

  std::vector<dtypeA> A_h(sizeA);
  std::vector<dtypeB> B_h(sizeB);
  std::vector<dtypeC> C_h(sizeC);
  std::vector<dtypeMeta> MetaA_h(sizeMetaA);
  std::vector<dtypeMeta> MetaB_h(sizeMetaB);

  std::generate(A_h.begin(), A_h.end(), [&]() {
    if constexpr (num_packed_a > 1) {
      return random_fp4_e3m0x2();
    } else {
      return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);;
    }
  });

  std::generate(B_h.begin(), B_h.end(),  [&]() {
    if constexpr (num_packed_b > 1) {
      return random_fp4_e3m0x2();
    } else {
      return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);;
    }
  });
  std::fill(C_h.begin(), C_h.end(), 0);

  std::generate(MetaA_h.begin(), MetaA_h.end(), []() { return random_scale(); });
  std::generate(MetaB_h.begin(), MetaB_h.end(), []() { return random_scale(); });

  auto A_d = malloc_device<dtypeA>(sizeA, q);
  auto B_d = malloc_device<dtypeB>(sizeB, q);
  auto C_d = malloc_device<dtypeC>(sizeC, q);
  auto MetaA_d = malloc_device<dtypeMeta>(sizeMetaA, q);
  auto MetaB_d = malloc_device<dtypeMeta>(sizeMetaB, q);

  q.memcpy(A_d, A_h.data(), sizeA * sizeof(dtypeA)).wait();
  q.memcpy(B_d, B_h.data(), sizeB * sizeof(dtypeB)).wait();
  q.memcpy(C_d, C_h.data(), sizeC * sizeof(dtypeC)).wait();
  q.memcpy(MetaA_d, MetaA_h.data(), sizeMetaA * sizeof(dtypeMeta)).wait();
  q.memcpy(MetaB_d, MetaB_h.data(), sizeMetaB * sizeof(dtypeMeta)).wait();

  constexpr uint32_t SubGroupSize = 32;
  constexpr uint32_t NumControlSubGroup = 4;
  constexpr uint32_t NumPostOpSubGroup = 16;
  range<3> local_size(1, NumControlSubGroup + NumPostOpSubGroup, SubGroupSize);
  range<3> group_size(1, ceil_div<size_t>(mat_m, wg_m), ceil_div<size_t>(mat_n, wg_n));
  nd_range<3> Range(group_size * local_size, local_size);

  static constexpr auto tnspA = (layout_a == mem_layout::row_major) ? xe4::GMMA::Major::K : xe4::GMMA::Major::MN;
  static constexpr auto tnspB = (layout_b == mem_layout::row_major) ? xe4::GMMA::Major::MN : xe4::GMMA::Major::K;

  using TileShape = Shape<Int<wg_m>, Int<wg_n>, Int<wg_k>>;
  using SplitTileShape = Shape<Int<wg_m>, Int<wg_n/split_b>, Int<wg_k>>;
  using TiledMma = decltype(cute::make_tiled_mma(cute::xe4::GMMA::ss_op_selector_scale<cute::xe4::GMMA::OpType::NoneCluster,
    dtypeA, dtypeB, dtypeAcc, dtypeC, dtypeMeta, SplitTileShape, tnspA, tnspB, a_blockscaling, b_blockscaling, mat_desc_t, meta_desc_t, abar_ptr_t>()));

  using LayoutA = std::conditional_t<tnspA == xe4::GMMA::Major::K, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>;
  using StrideA = cutlass::detail::TagToStrideA_t<LayoutA>;
  using LayoutB = std::conditional_t<tnspB == xe4::GMMA::Major::MN, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>;
  using StrideB = cutlass::detail::TagToStrideB_t<LayoutB>;

  using SmemLayoutAtomA = decltype(upcast<sizeof(dtypeA)>(make_layout(Shape<_32, _32>{}, GenRowMajor{})));
  using SmemLayoutAtomB = decltype(upcast<sizeof(dtypeB)>(make_layout(Shape<_32, _32>{}, std::conditional_t<tnspB == xe4::GMMA::Major::K, GenRowMajor, GenColMajor>{})));
  using SmemLayoutAtomC = Layout<Shape<Int<wg_m>, Int<wg_n>>, Stride<Int<wg_n>, _1>>;

  using CollectiveMainloop = CollectiveMma<
    MainloopXe4DmaGmmaWarpSpecializedMixedInput<stage, split_b, mx_scale_elem_num>, // Dispatch Policy
    TileShape,                                          // TileShape
    tuple<dtypeA,dtypeMeta>,                            // ElementATuple
    StrideA,                                            // StrideA
    tuple<dtypeB,dtypeMeta>,                            // ElementBTuple
    StrideB,                                            // StrideB
    TiledMma,                                           // TiledMma
    ASYNC_TENSOR_LOAD,                                  // GmemTiledCopyA
    SmemLayoutAtomA,                                    // SmemLayoutAtomA
    void,                                               // SmemCopyAtomA
    void,                                               // TransformA
    ASYNC_TENSOR_LOAD,                                  // GmemTiledCopyB
    SmemLayoutAtomB,                                    // SmemLayoutAtomB
    void,                                               // SmemCopyAtomB
    void                                                // TransformB
  >;

  static constexpr int FragmentSize = 2;
  using FusionOp = fusion::ScaledAcc<dtypeC, dtypeC>;
  using StrideC = cutlass::detail::TagToStrideC_t<cutlass::layout::RowMajor>;

  using FusionCallbacks = fusion::FusionCallbacks<
    Sm90TmaWarpSpecialized<1, 1, FragmentSize, false, false>,
    FusionOp, Shape<Int<wg_m>, Int<wg_n>, _1>, Shape<_2,_1>
  >;

  using CollectiveEpilogue = CollectiveEpilogue<
    Xe4DmaWarpSpecialized<FragmentSize, NumPostOpSubGroup, SubGroupSize>,
    dtypeC, StrideC, SmemLayoutAtomC, Shape<Int<wg_m>, Int<wg_n>>, FusionCallbacks
  >;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void
  >;

  using StrideScale = typename CollectiveMainloop::StrideScale;

  q.parallel_for<test>(Range, [=](nd_item<3> item) {
    auto problem_shape = make_tuple(mat_m, mat_n, mat_k, mat_l);

    auto layout_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(mat_m, mat_k, mat_l));
    auto layout_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(mat_n, mat_k, mat_l));
    auto layout_metaA = cutlass::make_cute_packed_stride(StrideScale{}, cute::make_shape(mat_m, meta_k, mat_l));
    auto layout_metaB = cutlass::make_cute_packed_stride(StrideScale{}, cute::make_shape(mat_n, meta_k, mat_l));

    auto args = typename GemmKernel::Arguments {
      {SubGroupSize, NumControlSubGroup, NumPostOpSubGroup},
      problem_shape,
      {
        A_d, layout_A, B_d, layout_B, MetaA_d, layout_metaA, MetaB_d, layout_metaB, meta_k
      },
      {
        typename FusionCallbacks::Arguments{},
        C_d, cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(mat_m, mat_n, mat_l)),
      }
    };

    GemmKernel kernel;
    auto params = kernel.to_underlying_arguments(args, nullptr);
    kernel(params, item);
  }).wait();

  q.memcpy(C_h.data(), C_d, sizeC * sizeof(dtypeC)).wait();

  free(A_d, q);
  free(B_d, q);
  free(C_d, q);
  free(MetaA_d, q);
  free(MetaB_d, q);

  int err_cnt = validate_mxfp_gemm_result(A_h.data(), B_h.data(), C_h.data(), mat_m, mat_n, mat_k, a_blockscaling, b_blockscaling, MetaA_h.data(), MetaB_h.data(), layout_a, layout_b);

  if (err_cnt > 0) {
    throw std::runtime_error("Test Failed!");
  } else {
    std::cout << "Test Pass!" << std::endl;
  }
}

int main() {
  run_test<GEMM_BF16_BF8_FP4>();
  run_test<GEMM_BF16_FP4_FP4>();
  run_test<GEMM_BF16_BF8_BF8>();
  return 0;
}
