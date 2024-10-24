#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/atom/copy_traits_xe4_dma.hpp"
#include <CL/sycl.hpp>
#include "inline_pisa.hpp"
#include "validation.hpp"
#include <cutlass/pipeline/xe4_pipeline.hpp>
#include "cutlass/conv/convnd_problem_shape.hpp"
#include "cutlass/conv/detail.hpp"

using namespace sycl;
using namespace cutlass::xe4;
using namespace cute;
using namespace cute::detail;
using namespace cute::xe4;

using ProblemShape = cutlass::conv::ConvProblemShape<cutlass::conv::Operator::kFprop, 2>;

class CONV2D_SMALL;
class CONV2D_LARGE;
class CONV2D_SMALL_WITH_PAD_WITH_STRIDE;
class CONV2D_LARGE_WITH_PAD_WITH_STRIDE;
class CONV2D_OTHER_WITH_PAD_WITH_STRIDE;
class CONV2D_ASYNMMETRIC_PAD_ASYNMMETRIC_STRIDE;
class CONV2D_LARGE_WITH_PAD_WITH_STRIDE_WITH_DILATION;
class CONV2D_OTHER_WITH_PAD_WITH_STRIDE_WITH_DILATION;
class CONV2D_ASYNMMETRIC_PAD_ASYNMMETRIC_STRIDE_WITH_DILATION;

static constexpr auto
get_problem_shape_MNKL(ProblemShape const& problem_shape) {
    return cutlass::conv::detail::get_linearized_problem_shape_MNKL(problem_shape);
}

// Activation cutlass::layout::TensorNHWC -> rank-2 stride ((W,H,N),_1)
template <class IntT>
CUTLASS_HOST_DEVICE
cute::Stride<cute::Stride<IntT, IntT, IntT>, cute::Int<1>>
make_cute_packed_stride(
    cute::Stride<cute::Stride<IntT, IntT, IntT>, cute::Int<1>> s,
    cute::array<IntT, 4> stride_nhwc) {
  static_assert(std::is_integral_v<IntT>,
    "Stride must have an integral type so it can be set dynamically. Static strides not supported.");
  assert(stride_nhwc[3] == 1);
  auto s_copy = s;
  cute::for_each(cute::make_seq<3>{}, [&](auto i) {
    cute::get<0,i>(s_copy) = stride_nhwc[2-i];
  });
  return s_copy;
}

// Filter cutlass::layout::TensorNHWC -> rank-2 stride (k, (_1, s, r))
template <class IntT>
CUTLASS_HOST_DEVICE
cute::Stride<IntT, cute::Stride<cute::Int<1>, IntT, IntT>>
make_cute_packed_stride(
    cute::Stride<IntT, cute::Stride<cute::Int<1>, IntT, IntT>> s,
    cute::array<IntT, 4> stride_krsc) {
  static_assert(std::is_integral_v<IntT>,
    "Stride must have an integral type so it can be set dynamically. Static strides not supported.");

  assert(stride_krsc[3] == 1);
  auto s_copy = s;
  cute::get<0,0>(s_copy) = stride_krsc[0];
  cute::for_each(cute::make_seq<2>{}, [&](auto i) {
    cute::get<1,2-i>(s_copy) = stride_krsc[i+1];
  });
  return s_copy;
}

// Compute the lower/near corner, returning it as a cute::array in [W,H] order
template <int NumSpatialDimensions>
CUTLASS_HOST_DEVICE
constexpr auto
compute_lower_corner_whd(cute::array<int32_t, NumSpatialDimensions> const& lower_padding) {
  cute::array<int, NumSpatialDimensions> lower{};

  for_each(make_seq<NumSpatialDimensions>{}, [&](auto i) {
    lower[NumSpatialDimensions-1-i] = -1 * lower_padding[i];
  });
 
  return lower;
}

// Computes the upper/far corner, returning it as a cute::array in [W,H] order
template <int NumSpatialDimensions>
CUTLASS_HOST_DEVICE
constexpr auto
compute_upper_corner_whd(cute::array<int32_t, NumSpatialDimensions> const& upper_padding, cute::array<int32_t, NumSpatialDimensions + 2> const& shape_B, cute::array<int32_t, NumSpatialDimensions> const& dilation) {
  cute::array<int, NumSpatialDimensions> upper{};
  cute::for_each(cute::make_seq<NumSpatialDimensions>{}, [&](auto i) {
    upper[NumSpatialDimensions-1-i] = upper_padding[i] - (shape_B[i+1] - 1) * dilation[i];
  });

  return upper;
}

// Compute the lower/near corner of (r,s), returning it as a cute::array in [S,R] order
template <int NumSpatialDimensions>
CUTLASS_HOST_DEVICE
constexpr auto
compute_lower_srt() {
  cute::array<int, NumSpatialDimensions> lower{};
  cute::for_each(cute::make_seq<NumSpatialDimensions>{}, [&](auto i) {
      lower[NumSpatialDimensions-1-i] = 0;
  });

  return lower;
}

template <int NumSpatialDimensions>
CUTLASS_HOST_DEVICE
constexpr auto
compute_stride_srt(cute::array<int32_t, NumSpatialDimensions> const& dilation) {
  cute::array<int32_t, NumSpatialDimensions> stride_srt{};
  cute::for_each(cute::make_seq<NumSpatialDimensions>{}, [&](auto i) {
        stride_srt[i] = dilation[NumSpatialDimensions-1-i];
  });

  return stride_srt;
}

// create DMA im2col descriptor
template <class EngineA, class LayoutA, class TMALayout, class LowerCornerStride, class UpperCornerStride, class LowerPaddingStride, class UpperPaddingStride, class TraversalStride, class LowerSRTStride, class DilationStride>
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
  auto gemm_shapes_common = make_shape(
      transform_leaf(gemm_mn, [](auto s) {
        return conditional_return(cute::is_static<decltype(s)>{}, s, cutlass::FastDivmod(s));
      }),
      gemm_k);
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

////////////////////////////////////////////////////////////////////////////////////////////////////
/// ASYNC_ROW_LOAD: Initiates a async row copy from global memory to shared memory
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
inline uint32_t get_copy_size(const int32_t coord, const uint32_t shape, uint32_t width_2d) {
    uint32_t left_size = (shape - coord) * sizeof(T);
    uint32_t copy_size = left_size < width_2d ? left_size : width_2d;
    return copy_size;
}

struct ASYNC_ROW_LOAD
{
  template<class TS, class TG, class CMType, class NumBytesPerCopy, class Tensor>
  CUTE_HOST_DEVICE static void
  copy(CMType cm_type, NumBytesPerCopy width_2d, Tensor gmem, uint64_t const* abar_ptr, TS* slm_ptr, TG* gmem_ptr, int32_t crd0, int32_t crd1, int32_t crd2, int32_t crd3)
  {
    sycl::vec<uint32_t, 4> gmem_shape {shape<0>(gmem), shape<1>(gmem), shape<2>(gmem), shape<3>(gmem)};
    sycl::vec<uint32_t, 3> gmem_stride {stride<1>(gmem) * sizeof(TG), stride<2>(gmem) * sizeof(TG), stride<3>(gmem) * sizeof(TG)};

    bool is_coord_valid = (crd1 >= 0) && (crd1 < gmem_shape[1]);
    is_coord_valid = is_coord_valid && (crd2 >= 0) && (crd2 < gmem_shape[2]);
    is_coord_valid = is_coord_valid && (crd3 >= 0) && (crd3 < gmem_shape[3]);

    uint32_t offset = crd0 * sizeof(TG) + crd1 * gmem_stride[0] + crd2  * gmem_stride[1] + crd3  * gmem_stride[2];
    offset = is_coord_valid ? offset : 0;
    uint32_t copy_size = is_coord_valid ? get_copy_size<TG>(crd0, gmem_shape[0], width_2d) : 0;
    async_2d_tiled_load<CMType::value, NumBytesPerCopy::value>(slm_ptr, gmem_ptr, offset, copy_size, abar_ptr);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// ASYNC_ROW_STORE: Initiates a async row copy from shared memory to global memory
////////////////////////////////////////////////////////////////////////////////////////////////////
struct ASYNC_ROW_STORE
{
  template<class TS, class TG, class CMType, class NumBytesPerCopy, class Tensor>
  CUTE_HOST_DEVICE static void
  copy(CMType cm_type, NumBytesPerCopy width_2d, Tensor gmem, uint64_t const* abar_ptr, TS* slm_ptr, TG* gmem_ptr, int32_t crd0, int32_t crd1, int32_t crd2, int32_t crd3)
  {
    sycl::vec<uint32_t, 4> gmem_shape {shape<0>(gmem), shape<1>(gmem), shape<2>(gmem), shape<3>(gmem)};
    sycl::vec<uint32_t, 3> gmem_stride {stride<1>(gmem) * sizeof(TG), stride<2>(gmem) * sizeof(TG), stride<3>(gmem) * sizeof(TG)};

    bool is_coord_valid = (crd1 >= 0) && (crd1 < gmem_shape[1]);
    is_coord_valid = is_coord_valid && (crd2 >= 0) && (crd2 < gmem_shape[2]);
    is_coord_valid = is_coord_valid && (crd3 >= 0) && (crd3 < gmem_shape[3]);

    uint32_t offset = crd0 * sizeof(TG) + crd1 * gmem_stride[0] + crd2 * gmem_stride[1] + crd3 * gmem_stride[2];
    offset = is_coord_valid ? offset : 0;
    uint32_t copy_size = is_coord_valid ? get_copy_size<TG>(crd0, gmem_shape[0], width_2d) : 0;
    async_2d_tiled_store<CMType::value, NumBytesPerCopy::value>(slm_ptr, gmem_ptr, offset, copy_size, abar_ptr);
  }
};

template<typename test>
int run_test(const conv2d::problem_shape_t &problem_shape)
{
    queue q;
    auto dev = q.get_device();
    std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";
    auto ctxt = q.get_context();

    // input:   (N, H, W, C)
    // kernel:  (K, R, S, C)
    // output:  (Out_N, Out_H, Out_W, Out_C)
    uint32_t N = problem_shape.get_in_batch();
    uint32_t H = problem_shape.get_in_height();
    uint32_t W = problem_shape.get_in_width();
    uint32_t C = problem_shape.get_in_channel();
    uint32_t K = problem_shape.get_kernel_num();
    uint32_t R = problem_shape.get_kernel_height();
    uint32_t S = problem_shape.get_kernel_width();
    uint32_t Out_N = problem_shape.get_out_batch();
    uint32_t Out_H = problem_shape.get_out_height();
    uint32_t Out_W = problem_shape.get_out_width();
    uint32_t Out_C = problem_shape.get_out_channel();
    uint32_t padding_top = problem_shape.get_padding_top();
    uint32_t padding_left = problem_shape.get_padding_left();
    uint32_t padding_bottom = problem_shape.get_padding_bottom();
    uint32_t padding_right = problem_shape.get_padding_right();
    uint32_t stride_h = problem_shape.get_stride_h();
    uint32_t stride_w = problem_shape.get_stride_w();
    uint32_t dilation_h = problem_shape.get_dilation_h();
    uint32_t dilation_w = problem_shape.get_dilation_w();
    cute::array<int32_t, 2> lower_padding{(int32_t) padding_top, (int32_t) padding_left};
    cute::array<int32_t, 2> upper_padding{(int32_t) padding_bottom, (int32_t) padding_right};
    cute::array<int32_t, 2> traversal_stride{(int32_t) stride_h, (int32_t) stride_w};
    cute::array<int32_t, 2> dilation{(int32_t) dilation_h, (int32_t) dilation_w};
    cute::array<int32_t, 4> shape_B{(int32_t) K, (int32_t) R, (int32_t) S, (int32_t) C};
    cute::array<int32_t, 4> shape_A{(int32_t) N, (int32_t) H, (int32_t) W, (int32_t) C};
    cute::array<int64_t, 4> stride_A{(int64_t) H*W*C, (int64_t) W*C, (int64_t) C, 1};

    ProblemShape cutlass_problem_shape {
        cutlass::conv::Mode::kCrossCorrelation,
        {static_cast<int>(N), static_cast<int>(H), static_cast<int>(W), static_cast<int>(C)},   // nhwc
        {static_cast<int>(K), static_cast<int>(R), static_cast<int>(S), static_cast<int>(C)},   // krsc
        {static_cast<int>(padding_left), static_cast<int>(padding_bottom)}, // padding lower (pad_h, pad_w)
        {static_cast<int>(padding_right), static_cast<int>(padding_top)},   // padding upper (pad_h, pad_w)
        {static_cast<int>(stride_h), static_cast<int>(stride_w)},           // stride (stride_h, stride_w)
        {static_cast<int>(dilation_h), static_cast<int>(dilation_w)},       // dilation (dilation_h, dilation_w)
        1   // group
    };

    auto [MNKL_M, MNKL_N, MNKL_K, MNKL_L] = get_problem_shape_MNKL(cutlass_problem_shape);
    std::cout << "[M, N, K, L] = " << MNKL_M << ", " << MNKL_N << ", " << MNKL_K << ", " << MNKL_L << std::endl;

    using dtypeA = bf16;
    using dtypeB = bf16;
    using dtypeAcc = float;
    using dtypeC = bf16;

    constexpr uint32_t dim = 4;
    constexpr uint32_t stage = 3;

    auto bM = Int<64>{};
    auto bN = Int<256>{};
    auto bK = Int<128>{};
    auto cta_tiler = make_shape(bM, bN, bK);
    using TileShapeMNK = Shape<Int<bM>, Int<bN>, Shape<Int<bK>>>;

    using Pipeline = cutlass::xe4::PipelineTmaAsync<stage>;
    using PipelineStore = cutlass::xe4::PipelineTmaAsync<1, 1>;

    constexpr mem_layout layout_a = mem_layout::row_major;
    constexpr mem_layout layout_b = mem_layout::col_major;
    constexpr bool is_col_major_a = layout_a == mem_layout::col_major;
    constexpr bool is_col_major_b = layout_b == mem_layout::col_major;

    uint32_t sizeA = C * W * H * N;
    uint32_t sizeB = C * S * R * K;
    uint32_t sizeC = Out_C * Out_W * Out_H * Out_N;

    auto A_shared = malloc_shared<dtypeA>(sizeA, q);
    std::generate_n(A_shared, sizeA, [=]() { return static_cast<float>(rand()) / static_cast<float>(RAND_MAX); });

    auto B_shared = malloc_shared<dtypeB>(sizeB, q);
    std::generate_n(B_shared, sizeB, [=]() { return static_cast<float>(rand()) / static_cast<float>(RAND_MAX); });

    auto C_shared = malloc_shared<dtypeC>(sizeC, q);
    std::fill_n(C_shared, sizeC, dtypeC(0));

    uint32_t repeat_c = (C + bK - 1) / bK;
    uint32_t kloop = repeat_c * S * R;

    range<3> local_range(1, 1, 64);
    uint32_t mat_m = Out_W * Out_H * Out_N;
    uint32_t mat_n = K;
    uint32_t mat_k = kloop * bK;
    uint32_t group_range_m = (mat_m + bM - 1) / bM;
    uint32_t group_range_n = (mat_n + bN - 1) / bN;
    range<3> group_range(1, group_range_m, group_range_n);
    std::cout << "Group range: {" << 1 << ", " << group_range_m << ", " << group_range_n << "} \n";
    nd_range<3> Range(group_range * local_range, local_range);

    // matrix info preparation
    constexpr slm_matrix_type cm_typeA = slm_matrix_type::type1;
    constexpr slm_matrix_type cm_typeB = slm_matrix_type::type1;
    constexpr slm_matrix_type cm_typeC = slm_matrix_type::type1;

    constexpr uint32_t width_2dA = bK * sizeof(dtypeA);
    constexpr uint32_t width_2dC = bN * sizeof(dtypeC);
    static_assert(bM % LANESIZE == 0);
    constexpr uint32_t num_inst = bM / LANESIZE;
    constexpr uint32_t inst_sizeA = LANESIZE * bK;
    constexpr uint32_t inst_sizeC = LANESIZE * bN;

    using mat_desc_t = uint32_t;
    using abar_ptr_t = uint64_t*;
    using tdesc_ptr_t = uint64_t*;
    constexpr uint32_t cm_bytes = 1024;

    using StrideA = decltype(cute::Stride<cute::Stride<int64_t, int64_t, int64_t>,cute::Int<1>>{});
    using StrideB = decltype(cute::Stride<int64_t, cute::Stride<cute::Int<1>, int64_t, int64_t>>{});
    auto shape_A_orig = make_shape(cute::reverse(cute::take<0, 3>(shape_A)), shape_A[3]);
    auto dA = make_cute_packed_stride(StrideA{}, stride_A);
    auto tensor_cwhn = make_tensor(A_shared, make_layout(shape_A_orig, dA));

    // compute the upper and lower corners based on conv padding
    auto lower_corner_whd = compute_lower_corner_whd<2>(lower_padding);
    auto upper_corner_whd = compute_upper_corner_whd<2>(upper_padding, shape_B, dilation);
    auto lower_srt = compute_lower_srt<2>();
    auto stride_srt = compute_stride_srt<2>(dilation);

    using MmaShapeMK = Shape<Int<bM>, Int<bK>>;
    auto cta_tiler_MK = MmaShapeMK{};
    auto cta_v_tile = make_identity_layout(product_each(shape(tensor_cwhn))).compose(cta_tiler_MK);
    auto glayout_basis = make_identity_layout(product_each(shape(tensor_cwhn)));

    //slayout manipulation
    auto slayout = make_layout(MmaShapeMK{}, Stride<Int<bK>, _1>{});
    auto inv_smem_layout = right_inverse(slayout);
    auto sidx_to_gmode = coalesce(composition(cta_v_tile, inv_smem_layout));
    auto tma_layout_full = flatten(composition(glayout_basis, sidx_to_gmode));

    auto smem_rank = find_if(stride(tma_layout_full), [](auto e) {
      [[maybe_unused]] auto v = basis_value(e);
      return not is_constant<1,decltype(v)>{};
    });

    constexpr int smem_tma_rank = cute::min(int(smem_rank), 2);
    auto tma_layout_trunc = take<0,smem_tma_rank>(tma_layout_full);
    auto tma_layout_vt = logical_divide(tma_layout_trunc, shape_div(size(tma_layout_trunc), 1));

    auto range_c    = size<0,0>(tma_layout_vt);
    auto range_whn = size<0,1>(tma_layout_vt);
    Tensor gtensor_cwhn = make_tensor(tensor_cwhn.data(),
                                     flatten(make_layout(make_layout(basis_get(stride<0,0>(tma_layout_vt), tensor_cwhn.shape()),
                                                                     basis_get(stride<0,0>(tma_layout_vt), tensor_cwhn.stride())),
                                                         make_layout(basis_get(stride<0,1>(tma_layout_vt), tensor_cwhn.shape()),
                                                                     basis_get(stride<0,1>(tma_layout_vt), tensor_cwhn.stride())))));
    auto tma_tensor = make_im2col_tma_copy_desc(gtensor_cwhn, range_c, range_whn,
                                                tma_layout_vt, 
                                                shape(lower_corner_whd),
                                                shape(upper_corner_whd),
                                                cute::reverse(shape(lower_padding)),
                                                cute::reverse(shape(upper_padding)),
                                                cute::reverse(shape(traversal_stride)),
                                                shape(lower_srt),
                                                shape(stride_srt));

    #if 0
    print("tensor_cwhn          :"); print(tensor_cwhn.layout()); print("\n");
    print("lower_corner_whd     :"); print(lower_corner_whd); print("\n");
    print("upper_corner_whd     :"); print(upper_corner_whd); print("\n");
    print("lower_srt            :"); print(lower_srt); print("\n");
    print("stride_srt           :"); print(stride_srt); print("\n");
    print("glayout_basis        :"); print(glayout_basis); print("\n");
    print("slayout              :"); print(slayout); print("\n");
    print("cta_tiler            :"); print(cta_tiler); print("\n");
    print("cta_v_tile           :"); print(cta_v_tile); print("\n");
    print("inv_smem_layout      :"); print(inv_smem_layout); print("\n");
    print("sidx_to_gmode        :"); print(sidx_to_gmode); print("\n");
    print("tma_layout_full      :"); print(tma_layout_full); print("\n");
    print("tma_layout_vt        :"); print(tma_layout_vt); print("\n");
    print("gtensor_cwhn         :"); print(gtensor_cwhn.layout()); print("\n");
    print("tma_tensor           :"); print_tensor(tma_tensor); print("\n");
    #endif

    q.parallel_for<test>(Range, [=](nd_item<3> item) {
        uint32_t local_id = item.get_local_linear_id();
        uint32_t subgroup_id = local_id / 32;

        Pipeline pipeline(local_id);
        abar_ptr_t abar_cons_base = pipeline.abar_cons_base;

        PipelineStore pipeline_store(local_id);

        auto layoutA = make_layout(make_shape(C, W, H, N));
        auto layoutB = make_layout(make_shape(C, S, R, K));
        auto layoutC = make_layout(make_shape(Out_C, Out_W, Out_H, Out_N));
        auto A = make_tensor(A_shared, layoutA);
        auto B = make_tensor(B_shared, layoutB);
        auto C = make_tensor(C_shared, layoutC);

        auto layoutSA = make_layout(Shape<Int<bM>, Int<bK>, Int<stage>>{}, Stride<Int<bK>, _1, Int<bM * bK>>{});
        auto layoutSB = make_layout(Shape<Int<bN>, Int<bK>, Int<stage>>{}, Stride<Int<bK>, _1, Int<bN * bK>>{});
        auto layoutSAcc = make_layout(Shape<Int<bM>, Int<bN>>{}, Stride<Int<bN>, _1>{});
        auto layoutSC = make_layout(Shape<Int<bM>, Int<bN>>{}, Stride<Int<bN>, _1>{});

        constexpr uint32_t total_bytes_a = size(layoutSA) * sizeof(dtypeA);
        constexpr uint32_t total_bytes_b = size(layoutSB) * sizeof(dtypeB);
        constexpr uint32_t total_bytes_c = size(layoutSC) * sizeof(dtypeC);
        constexpr uint32_t total_bytes_acc = size(layoutSC) * sizeof(dtypeAcc);

        constexpr uint32_t slm_bytes_a = total_bytes_a / stage;
        constexpr uint32_t slm_bytes_b = total_bytes_b / stage;
        constexpr uint32_t slm_bytes_c = total_bytes_c;

        constexpr uint32_t slm_bytes = total_bytes_a + total_bytes_b + total_bytes_c + total_bytes_acc;
        auto slm_ptr = alloc_slm_buffer<uint8_t, slm_bytes>(item.get_group());

        auto sA = make_tensor(reinterpret_cast<dtypeA*>(slm_ptr), layoutSA);
        auto sB = make_tensor(reinterpret_cast<dtypeB*>(sA.data() + size(layoutSA)), layoutSB);
        auto sAcc = make_tensor(reinterpret_cast<dtypeAcc*>(sB.data() + size(layoutSB)), layoutSAcc);
        auto sC = make_tensor(reinterpret_cast<dtypeC*>(sAcc.data() + size(layoutSAcc)), layoutSC);

        uint32_t wg_id_x = item.get_group(2);
        uint32_t wg_id_y = item.get_group(1);

        int start_m = wg_id_y * bM;
        int start_n = wg_id_x * bN;
        auto cta_coord = make_coord(wg_id_y, wg_id_x, _);              // (m,n,k)

        item.barrier(access::fence_space::local_space);

        if(subgroup_id == 0){
            using AuxParamsB = AuxParams<cm_typeB, tdesc_ptr_t, 0>;
            // auto tdesc_ptrB = make_conv2d_tensor_desc<AuxParamsB>(B, layoutSB);
            using T = dtypeB;
  auto dim = rank(B);
  
  sycl::vec<uint32_t, dim> gmem_shape {shape<0>(B), shape<1>(B), shape<2>(B), shape<3>(B)};
  sycl::vec<uint64_t, dim - 1> gmem_stride {stride<1>(B) * sizeof(T), stride<2>(B) * sizeof(T),
    stride<3>(B) * sizeof(T)};
  sycl::vec<uint32_t, dim> roi_shape {shape<1>(layoutSB), 1, 1, shape<0>(layoutSB)};
  sycl::vec<uint32_t, dim> elem_stride {1, 1, 1, 1};

  auto tdesc_ptrB = allocate_tdesc<AuxParamsB::tdescIdx, typename AuxParamsB::tdescPtr>();
  tensor_desc_fill_global_addr(tdesc_ptrB, B.data());
  tensor_descriptor_fill_dim_size<dim>(tdesc_ptrB, gmem_shape);
  tensor_descriptor_fill_dim_stride<dim>(tdesc_ptrB, gmem_stride);
  tensor_descriptor_fill_traverse_stride<dim>(tdesc_ptrB, elem_stride);
  tensor_descriptor_fill_roitensor_size<dim>(tdesc_ptrB, roi_shape);
  tensor_descriptor_fill_misc<T, AuxParamsB::cmType>(tdesc_ptrB);


            auto mB_nk = make_counting_tensor(make_layout(shape(layoutB), make_stride(E<0>{}, E<1>{}, E<2>{}, E<3>{})));
            auto gB_nk = tiled_divide(mB_nk, make_tile(size<1>(layoutSB), Int<1>{}, Int<1>{}, size<0>(layoutSB)));
            // if (DEBUG_THREAD) {
            //   PRINT(mB_nk);
            //   PRINT(gB_nk);
            // }

            auto shape_B_orig = cutlass_problem_shape.get_shape_B();
            auto dB = make_cute_packed_stride(StrideB{}, cutlass_problem_shape.stride_B);
            Tensor tensor_b = make_tensor(B_shared, make_layout(shape_B_orig, dB));
            // auto tma_load_b = get_tma_load_b_instance(tensor_b, cutlass_problem_shape);

            // auto dB = make_cute_packed_stride(StrideB{}, problem_shape.stride_B, ConvOp);
            

            auto tilerB = Shape<Int<bN>, Shape<Int<bK>>>{};
            auto tmp_layoutSB = make_layout(Shape<Int<bN>, Int<bK>>{}, Stride<Int<bK>, _1>{});
        
            auto load_b = make_xe4_copy_conv2d<ASYNC_TENSOR_LOAD, AuxParamsB>(tensor_b, tmp_layoutSB, tilerB);
            // auto load_b = make_xe4_copy<ASYNC_TENSOR_LOAD, AuxParamsB>(B, layoutSB, tilerB);

            // Tensor tmp_mB_nk = load_b.get_tma_tensor(make_shape(MNKL_N,MNKL_K));
            // Tensor tmp_gB_nk = local_tile(tmp_mB_nk, TileShapeMNK{}, make_coord(_,_,_), Step< X,_1,_1>{});

            if (DEBUG_THREAD) {
              PRINT(shape_B_orig);
              PRINT(dB);
              PRINT(tensor_b);
              PRINT(tmp_layoutSB);
              PRINT(tilerB);
              PRINT(load_b);
              // PRINT(tmp_mB_nk);
              // PRINT(tmp_gB_nk);
            }
            

            sycl::vec<uint32_t, dim> gmem_shapeC {shape<0>(C), shape<1>(C), shape<2>(C), shape<3>(C)};
            sycl::vec<uint32_t, dim - 1> gmem_strideC {stride<1>(C) * sizeof(dtypeC),
                stride<2>(C) * sizeof(dtypeC),
                stride<3>(C) * sizeof(dtypeC)};
            sycl::vec<uint32_t, dim> roi_shapeC {shape<1>(layoutSC), shape<1>(C), shape<2>(C), shape<3>(C)};

            sycl::vec<uint32_t, dim> gmem_shapeA {shape<0>(A), shape<1>(A), shape<2>(A), shape<3>(A)};
            sycl::vec<uint32_t, dim - 1> gmem_strideA {stride<1>(A) * sizeof(dtypeA),
                stride<2>(A) * sizeof(dtypeA),
                stride<3>(A) * sizeof(dtypeA)};
            sycl::vec<uint32_t, dim> roi_shapeA {shape<1>(layoutSA), shape<1>(C), shape<2>(C), shape<3>(C)};

            int32_t coord_offset_m_base = start_m + local_id;
            int32_t coord_table[num_inst * (dim - 1)];
            #pragma unroll
            for (uint32_t inst_idx = 0; inst_idx < num_inst; inst_idx++) {
                uint32_t index = inst_idx * (dim - 1);
                int32_t offset_m = coord_offset_m_base;

                coord_table[index] = offset_m % roi_shapeA[1];
                offset_m = offset_m / roi_shapeA[1];
                coord_table[index + 1] = offset_m % roi_shapeA[2];
                coord_table[index + 2] = offset_m / roi_shapeA[2];

                coord_offset_m_base += LANESIZE;
            }

            auto slm_pipe_write = cutlass::xe4::make_producer_start_state<Pipeline>();
            auto k_tile_iter = cute::make_coord_iterator(make_shape(repeat_c, S, R));
            auto k_tile_count = size(make_shape(repeat_c, S, R));
            for ( ; k_tile_count > 0; --k_tile_count) {
                uint32_t iter0 = get<0>(*k_tile_iter);
                uint32_t iter1 = get<1>(*k_tile_iter);
                uint32_t iter2 = get<2>(*k_tile_iter);

                uint32_t abar_index = slm_pipe_write.index();
                auto abar_prod = pipeline.producer_get_barrier(abar_index);

                pipeline.producer_try_wait(slm_pipe_write);

                // load input with row_copy
                int32_t gmem_coord_base0 = iter0 * bK;
                int32_t gmem_coord_base1 = iter1 * dilation_w;
                int32_t gmem_coord_base2 = iter2 * dilation_h;

                auto tA = sA(_, _, abar_index);
                auto slm_ptr_a = slm_space_cast(tA.data());

                #pragma unroll
                for (uint32_t inst_idx = 0; inst_idx < num_inst; inst_idx++) {
                    auto inst_slm_ptr_a = slm_ptr_a + inst_idx * inst_sizeA;
                    uint32_t index = inst_idx * (dim - 1);
                    auto coord_offset1 = coord_table[index] * stride_w - padding_left;
                    auto coord_offset2 = coord_table[index + 1] * stride_h - padding_top;
                    auto coord_offset3 = coord_table[index + 2];

                    ASYNC_ROW_LOAD::copy(cute::C<cm_typeA>{}, cute::C<width_2dA>{}, A, abar_prod, inst_slm_ptr_a, A_shared, 
                                         gmem_coord_base0, gmem_coord_base1 + coord_offset1, gmem_coord_base2 + coord_offset2, coord_offset3);
                }

                if(local_id == 0) {
                    pipeline.producer_commit(abar_index, slm_bytes_a + slm_bytes_b);

                    // load kernel with tensor_copy
                    auto tB = sB(_, _, abar_index);
                    auto gmem_coord = gB_nk(0, iter0, iter1, iter2, wg_id_x);
                    ASYNC_TENSOR_LOAD::copy(tdesc_ptrB, abar_prod, tB.data(), get<0>(gmem_coord), get<1>(gmem_coord),
                        get<2>(gmem_coord), get<3>(gmem_coord));
                }

                ++k_tile_iter;
                ++slm_pipe_write;
            }

            //store out
            PipelineStore::PipelineState slm_pipe_store_cons;
            pipeline_store.consumer_try_wait(slm_pipe_store_cons);

            if(local_id == 0) {
                pipeline_store.consumer_commit(slm_pipe_store_cons, slm_bytes_c);
            }

            auto slm_ptr_c = slm_space_cast(sC.data());

            #pragma unroll
            for (uint32_t inst_idx = 0; inst_idx < num_inst; inst_idx++) {
                auto inst_slm_ptr_c = slm_ptr_c + inst_idx * inst_sizeC;
                uint32_t index = inst_idx * (dim - 1);

                uint32_t abar_store_cons_index = slm_pipe_store_cons.index();
                auto abar_store_cons = pipeline_store.producer_get_barrier(abar_store_cons_index);
                ASYNC_ROW_STORE::copy(cute::C<cm_typeC>{}, cute::C<width_2dC>{}, C, abar_store_cons, inst_slm_ptr_c, C_shared, start_n, coord_table[index], coord_table[index+1], coord_table[index+2]);
            }

            ++slm_pipe_store_cons;
            pipeline_store.producer_try_wait(slm_pipe_store_cons);
        }
        else if (subgroup_id == 1){
            if (local_id == 32){
                mat_desc_t mat_desc_a = reinterpret_cast<uint64_t>(slm_space_cast(sA.data())) >> 9;
                mat_desc_t mat_desc_b = reinterpret_cast<uint64_t>(slm_space_cast(sB.data())) >> 9;
                mat_desc_t mat_desc_acc = reinterpret_cast<uint64_t>(slm_space_cast(sAcc.data())) >> 9;
                mat_desc_t mat_desc_c = reinterpret_cast<uint64_t>(slm_space_cast(sC.data())) >> 9;
                constexpr uint32_t cm_size_a_x = is_col_major_a ? 32: 32 / sizeof(dtypeA);
                constexpr uint32_t cm_num_a_x = is_col_major_a ? bM / cm_size_a_x : bK / cm_size_a_x;
                constexpr uint32_t cm_size_b_x = 32 / sizeof(dtypeB);
                constexpr uint32_t cm_num_b_x = is_col_major_b ? bK / cm_size_b_x : bN / cm_size_b_x;
                constexpr uint32_t cm_size_c_x = 32 / sizeof(dtypeC);
                constexpr uint32_t cm_num_c_x = bN / cm_size_c_x;
                constexpr uint32_t cm_size_acc_x = 32 / sizeof(dtypeAcc);
                constexpr uint32_t cm_num_acc_x = bN / cm_size_acc_x;
                constexpr uint32_t cm_stride_a = (cm_bytes * cm_num_a_x) >> 10;
                constexpr uint32_t cm_stride_b = (cm_bytes * cm_num_b_x) >> 10;
                constexpr uint32_t cm_stride_c = (cm_bytes * cm_num_c_x) >> 10;
                constexpr uint32_t cm_stride_acc = (cm_bytes * cm_num_acc_x) >> 10;
                mat_desc_a |= (cm_stride_a << 16);
                mat_desc_b |= (cm_stride_b << 16);
                mat_desc_c |= (cm_stride_c << 16);
                mat_desc_acc |= (cm_stride_acc << 16);

                if (kloop == 1) {
                    Pipeline::PipelineState slm_pipe_read;
                    pipeline.consumer_try_wait(slm_pipe_read);
                    async_gmma<dtypeC, dtypeA, dtypeB, bM, bN, bK, layout_a, layout_b>(mat_desc_c, mat_desc_a, mat_desc_b, abar_cons_base);
                    pipeline.consumer_commit(slm_pipe_read);
                } else {
                    Pipeline::PipelineState slm_pipe_read;
                    pipeline.consumer_try_wait(slm_pipe_read);
                    async_gmma<dtypeAcc, dtypeA, dtypeB, bM, bN, bK, layout_a, layout_b>(
                                mat_desc_acc, mat_desc_a, mat_desc_b, abar_cons_base);
                    pipeline.consumer_commit(slm_pipe_read);

                    static_assert(stage > 1);
                    for (uint32_t i = 1; i < kloop - 1; i++) {
                        ++slm_pipe_read;
                        uint32_t abar_index = slm_pipe_read.index();
                        auto abar_cons = pipeline.consumer_get_barrier(abar_index);
                        auto slm_offset_a = (abar_index * slm_bytes_a) >> 9;
                        auto slm_offset_b = (abar_index * slm_bytes_b) >> 9;
                        pipeline.consumer_try_wait(slm_pipe_read);

                        async_gmma<dtypeAcc, dtypeAcc, dtypeA, dtypeB, bM, bN, bK, layout_a, layout_b>(
                                    mat_desc_acc, mat_desc_acc, mat_desc_a + slm_offset_a, mat_desc_b + slm_offset_b,
                                    abar_cons);
                        pipeline.consumer_commit(slm_pipe_read);
                    }
                    {
                        auto slm_pipe_store_prod = cutlass::xe4::make_producer_start_state<PipelineStore>();
                        uint32_t abar_store_prod_index = slm_pipe_store_prod.index();
                        auto abar_store_prod = pipeline_store.producer_get_barrier(abar_store_prod_index);

                        ++slm_pipe_read;
                        uint32_t abar_index = slm_pipe_read.index();
                        auto slm_offset_a = (abar_index * slm_bytes_a) >> 9;
                        auto slm_offset_b = (abar_index * slm_bytes_b) >> 9;

                        auto abar_cons = pipeline.consumer_get_barrier(abar_index);
                        auto abar_prod = pipeline.producer_get_barrier(abar_index);

                        uint32_t phase = ((kloop - 1) / stage) & 1u;
                        pipeline.consumer_try_wait(abar_index, phase);

                        async_gmma<dtypeC, dtypeAcc, dtypeA, dtypeB, bM, bN, bK, layout_a, layout_b>(
                            mat_desc_c, mat_desc_acc, mat_desc_a + slm_offset_a, mat_desc_b + slm_offset_b, abar_store_prod);
                        pipeline_store.producer_commit(slm_pipe_store_prod, 1);
                    }
                }
            }
        }
    }).wait();

    uint32_t err_cnt = validate_conv2d_result_by_onednn(A_shared, B_shared, C_shared, problem_shape);

    int rtn = 0;
    if (err_cnt > 0)
    {
        std::cout << "Test Failed!" << std::endl;
        rtn = -1;
    }
    else
    {
        std::cout << "Test Pass!" << std::endl;
        rtn = 0;
    }

    return rtn;
}

int main(){
#if defined(TEST_SMALL)
    conv2d::problem_shape_t small {{64, 8, 8, 2}, {64, 3, 3, 128}, {0, 0}, {0, 0}, {1, 1}, {1, 1}};
    run_test<CONV2D_SMALL>(small);
#elif defined(TEST_LARGE)
    conv2d::problem_shape_t large {{160, 16, 16, 2}, {160, 3, 3, 224}, {0, 0}, {0, 0}, {1, 1}, {1, 1}};
    run_test<CONV2D_LARGE>(large);
#elif defined(TEST_SMALL_WITH_PAD_WITH_STRIDE)
    conv2d::problem_shape_t small_with_pad_with_stride {{80, 7, 7, 1}, {80, 3, 3, 80}, {1, 1}, {1, 1}, {2, 2}, {1, 1}};
    run_test<CONV2D_SMALL_WITH_PAD_WITH_STRIDE>(small_with_pad_with_stride);
#elif defined(TEST_LARGE_WITH_PAD_WITH_STRIDE)
    conv2d::problem_shape_t large_with_pad_with_stride {{160, 16, 16, 2}, {160, 3, 3, 224}, {1, 1}, {1, 1}, {2, 2}, {1, 1}};
    run_test<CONV2D_LARGE_WITH_PAD_WITH_STRIDE>(large_with_pad_with_stride);
#elif defined(TEST_OTHER_WITH_PAD_WITH_STRIDE)
    conv2d::problem_shape_t other_with_pad_with_stride {{160, 16, 16, 2}, {160, 5, 5, 224}, {1, 1}, {1, 1}, {3, 3}, {1, 1}};
    run_test<CONV2D_OTHER_WITH_PAD_WITH_STRIDE>(other_with_pad_with_stride);
#elif defined(TEST_ASYNMMETRIC_PAD_ASYNMMETRIC_STRIDE)
    conv2d::problem_shape_t asymmetric_pad_asynmmetric_stride {{160, 16, 16, 2}, {160, 3, 3, 224}, {1, 2}, {3, 4}, {2, 3}, {1, 1}};
    run_test<CONV2D_ASYNMMETRIC_PAD_ASYNMMETRIC_STRIDE>(asymmetric_pad_asynmmetric_stride);
#elif defined(TEST_LARGE_WITH_PAD_WITH_STRIDE_WITH_DILATION)
    conv2d::problem_shape_t large_with_pad_with_stride_with_dilation {{160, 16, 16, 2}, {160, 3, 3, 224}, {1, 1}, {1, 1}, {2, 2}, {2, 2}};
    run_test<CONV2D_LARGE_WITH_PAD_WITH_STRIDE_WITH_DILATION>(large_with_pad_with_stride_with_dilation);  // generated XeISA is for this casae
#elif defined(TEST_OTHER_WITH_PAD_WITH_STRIDE_WITH_DILATION)
        conv2d::problem_shape_t other_with_pad_with_stride_with_dilation {{160, 16, 16, 2}, {160, 5, 5, 224}, {1, 1}, {1, 1}, {3, 3}, {3, 3}};
        run_test<CONV2D_OTHER_WITH_PAD_WITH_STRIDE_WITH_DILATION>(other_with_pad_with_stride_with_dilation);
#elif defined(TEST_ASYNMMETRIC_PAD_ASYNMMETRIC_STRIDE_WITH_DILATION)
        conv2d::problem_shape_t asymmetric_pad_asynmmetric_stride_with_dilation {{160, 16, 16, 2}, {160, 3, 3, 224}, {1, 2}, {3, 4}, {2, 3}, {2, 3}};
        run_test<CONV2D_ASYNMMETRIC_PAD_ASYNMMETRIC_STRIDE_WITH_DILATION>(asymmetric_pad_asynmmetric_stride_with_dilation);
#endif

    return 0;
}
