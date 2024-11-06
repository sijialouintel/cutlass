#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/atom/copy_traits_xe4_dma.hpp"
#include "cute/atom/mma_traits_xe4_amma.hpp"
#include "cute/atom/copy_traits_xe4_im2col.hpp"
#include <CL/sycl.hpp>
#include "inline_pisa.hpp"
#include "validation.hpp"
#include <cutlass/pipeline/xe4_pipeline.hpp>
#include "cutlass/conv/convnd_problem_shape.hpp"
#include "cutlass/conv/detail.hpp"
#include "cute/arch/mma_xe4.hpp"
#include "cute/arch/mma_xe4_amma.hpp"

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
  // static_assert(std::is_integral_v<IntT>,
  //   "Stride must have an integral type so it can be set dynamically. Static strides not supported.");
  // assert(stride_nhwc[3] == 1);
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
  // static_assert(std::is_integral_v<IntT>,
  //   "Stride must have an integral type so it can be set dynamically. Static strides not supported.");

  // assert(stride_krsc[3] == 1);
  auto s_copy = s;
  cute::get<0,0>(s_copy) = stride_krsc[0];
  cute::for_each(cute::make_seq<2>{}, [&](auto i) {
    cute::get<1,2-i>(s_copy) = stride_krsc[i+1];
  });
  return s_copy;
}

template <int NumSpatialDimensions>
CUTLASS_HOST_DEVICE
constexpr auto
compute_lower_corner_whd(ProblemShape const& problem_shape) {
  using cute::for_each;
  using cute::make_seq;

  cute::array<int, NumSpatialDimensions> lower{};
  for_each(make_seq<NumSpatialDimensions>{}, [&](auto i) {
    lower[NumSpatialDimensions-1-i] = -1 * problem_shape.lower_padding[i];
  });

  return lower;
}

// Computes the upper/far corner, returning it as a cute::array in [W,H,D] order
template <int NumSpatialDimensions>
CUTLASS_HOST_DEVICE
constexpr auto
compute_upper_corner_whd(ProblemShape const& problem_shape) {
  using cute::for_each;
  using cute::make_seq;

  cute::array<int, NumSpatialDimensions> upper{};
  for_each(make_seq<NumSpatialDimensions>{}, [&](auto i) {
    upper[NumSpatialDimensions-1-i] = problem_shape.upper_padding[i] -
      (problem_shape.shape_B[i+1] - 1) * problem_shape.dilation[i];
  });

  return upper;
}


// Compute the lower/near corner of (r,s), returning it as a cute::array in [S,R] order
template <int NumSpatialDimensions>
CUTLASS_HOST_DEVICE
constexpr auto
compute_lower_srt(ProblemShape const& problem_shape) {
  using cute::for_each;
  using cute::make_seq;

  cute::array<int, NumSpatialDimensions> lower{};
  for_each(make_seq<NumSpatialDimensions>{}, [&](auto i) {
    lower[NumSpatialDimensions-1-i] = 0;
  });
  return lower;
}

template <int NumSpatialDimensions>
CUTLASS_HOST_DEVICE
constexpr auto
compute_stride_srt(ProblemShape const& problem_shape) {
  cute::array<int32_t, NumSpatialDimensions> stride_srt{};
  cute::for_each(cute::make_seq<NumSpatialDimensions>{}, [&](auto i) {
        stride_srt[i] = problem_shape.dilation[NumSpatialDimensions-1-i];
  });

  return stride_srt;
}

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

    using dtypeA = bf16;
    using dtypeB = bf16;
    using dtypeAcc = float;
    using dtypeC = bf16;

    constexpr uint32_t dim = 4;
    constexpr uint32_t stage = 3;

    auto bM = Int<64>{};
    auto bN = Int<256>{};
    auto bK = Int<128>{};
    using TileShapeMNK = Shape<Int<bM>, Int<bN>, Shape<Int<bK>>>;
    using MmaShapeMK = Shape<Int<bM>, Int<bK>>;
    using MmaShapeMN = Shape<Int<bM>, Int<bN>>;
    using MmaTiler = Shape<Int<bM>, Int<bN>, Int<bK>>;

    using Pipeline = cutlass::xe4::PipelineTmaAsync<stage>;
    using PipelineStore = cutlass::xe4::PipelineTmaAsync<1, 1>;

    constexpr mem_layout layout_a = mem_layout::row_major;
    constexpr mem_layout layout_b = mem_layout::col_major;
    constexpr bool is_row_major_a = layout_a == mem_layout::row_major;
    constexpr bool is_row_major_b = layout_b == mem_layout::row_major;

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
    constexpr uint32_t num_bits_per_tma = width_2dA * 8;
    constexpr uint32_t num_bits_per_tma_store = width_2dC * 8;
    static_assert(bM % LANESIZE == 0);
    constexpr uint32_t num_inst = bM / LANESIZE;
    constexpr uint32_t inst_sizeA = LANESIZE * bK;
    constexpr uint32_t inst_sizeC = LANESIZE * bN;

    using MatDesc = uint32_t;
    using Abarrier = uint64_t*;
    using tdesc_ptr_t = uint64_t*;

    using AuxParamsB = AuxParams<cm_typeB, tdesc_ptr_t, 0>;
    using StrideA = decltype(cute::Stride<cute::Stride<int64_t, int64_t, int64_t>,cute::Int<1>>{});
    using StrideB = decltype(cute::Stride<int64_t, cute::Stride<cute::Int<1>, int64_t, int64_t>>{});
    using StrideC = decltype(cute::Stride<cute::Stride<int64_t, int64_t, int64_t>,cute::Int<1>>{});

    q.parallel_for<test>(Range, [=](nd_item<3> item) {
        uint32_t local_id = item.get_local_linear_id();
        uint32_t subgroup_id = local_id / 32;

        Pipeline pipeline(local_id);
        Abarrier abar_cons_base = pipeline.abar_cons_base;

        PipelineStore pipeline_store(local_id);

        ProblemShape cutlass_problem_shape {
            cutlass::conv::Mode::kCrossCorrelation,
            {static_cast<int>(N), static_cast<int>(H), static_cast<int>(W), static_cast<int>(C)},   // nhwc
            {static_cast<int>(K), static_cast<int>(R), static_cast<int>(S), static_cast<int>(C)},   // krsc
            {static_cast<int>(padding_top), static_cast<int>(padding_left)}, // padding lower (pad_top, pad_left)
            {static_cast<int>(padding_bottom), static_cast<int>(padding_right)},   // padding upper (pad_bottom, pad_right)
            {static_cast<int>(stride_h), static_cast<int>(stride_w)},           // stride (stride_h, stride_w)
            {static_cast<int>(dilation_h), static_cast<int>(dilation_w)},       // dilation (dilation_h, dilation_w)
            1   // group
        };
        auto [MNKL_M, MNKL_N, MNKL_K, MNKL_L] = get_problem_shape_MNKL(cutlass_problem_shape);

        auto layoutC = make_layout(make_shape(Out_C, Out_W, Out_H, Out_N));
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

        item.barrier(access::fence_space::local_space);

        // todo: hard code
        auto layoutAtomA = make_layout(Shape<Int<32>, Int<16>>{}, Stride<Int<16>, Int<1>>{});
        auto layoutSA_mma = tile_to_shape(layoutAtomA, layoutSA.shape(), Step<_2,_1,_3>{});
        auto sA_mma = make_tensor(reinterpret_cast<dtypeA*>(slm_ptr), layoutSA_mma);

        auto layoutAtomB = make_layout(Shape<Int<32>, Int<16>>{}, Stride<Int<16>, Int<1>>{});
        auto layoutSB_mma = tile_to_shape(layoutAtomB, layoutSB.shape(), Step<_2,_1,_3>{});
        auto sB_mma = make_tensor(reinterpret_cast<dtypeB*>(sA.data() + size(layoutSA)), layoutSB_mma);

        auto layoutAtomC = make_layout(Shape<Int<32>, Int<16>>{}, Stride<Int<16>, Int<1>>{});
        auto layoutSC_mma = tile_to_shape(layoutAtomC, layoutSC.shape(), Step<_2,_1>{});
        auto sC_mma = make_tensor(reinterpret_cast<dtypeC*>(sAcc.data() + size(layoutSAcc)), layoutSC_mma);

        auto layoutAtomAcc = make_layout(Shape<Int<32>, Int<8>>{}, Stride<Int<8>, Int<1>>{});
        auto layoutSAcc_mma = tile_to_shape(layoutAtomAcc, layoutSAcc.shape(), Step<_2,_1>{});
        auto sAcc_mma = make_tensor(reinterpret_cast<dtypeAcc*>(sB.data() + size(layoutSB)), layoutSAcc_mma);

        constexpr auto scaleOutOne = cute::C<cute::xe4::AMMA::ScaleOut::One>{};
        constexpr auto scaleOutZero = cute::C<cute::xe4::AMMA::ScaleOut::Zero>{};

        using TiledMmaQuaternion = decltype(cute::make_tiled_mma(AMMA::ss_op_selector<AMMA::OpType::NoneCluster, dtypeA, dtypeB, dtypeAcc, dtypeC, MmaTiler, is_row_major_a, is_row_major_b>()));
        using TiledMmaTernary = decltype(cute::make_tiled_mma(AMMA::ss_op_selector<AMMA::OpType::NoneCluster, dtypeA, dtypeB, dtypeAcc, MmaTiler, is_row_major_a, is_row_major_b>()));
        TiledMmaQuaternion tiled_mma_quaternion;
        TiledMmaTernary tiled_mma_ternary;
        auto thread_mma_quaternion = tiled_mma_quaternion.get_thread_slice(local_id - 32);

        auto tCrA = thread_mma_quaternion.partition_fragment_A(sA_mma);            // (MMA,MMA_M,MMA_K,PIPE)
        auto tCrB = thread_mma_quaternion.partition_fragment_B(sB_mma);            // (MMA,MMA_N,MMA_K,PIPE)
        auto accum = thread_mma_quaternion.partition_fragment_C(sAcc_mma);         // (MMA,MMA_M,MMA_N)
        auto tCrC = thread_mma_quaternion.partition_fragment_C(sC_mma);            // (MMA,MMA_M,MMA_N)

        if(subgroup_id == 0){
            auto shape_A_orig = make_shape(cute::reverse(cute::take<0, 3>(cutlass_problem_shape.shape_A)), cutlass_problem_shape.shape_A[3]);
            auto dA = make_cute_packed_stride(StrideA{}, cutlass_problem_shape.stride_A);
            auto tensor_cwhn = make_tensor(A_shared, make_layout(shape_A_orig, dA));

            auto shape_B_orig = cutlass_problem_shape.get_shape_B();
            auto dB = make_cute_packed_stride(StrideB{}, cutlass_problem_shape.stride_B);
            Tensor gtensorB = make_tensor(B_shared, make_layout(shape_B_orig, dB));

            auto shape_C_orig = make_shape(cute::reverse(cute::take<0, 3>(cutlass_problem_shape.shape_C)), cutlass_problem_shape.shape_C[3]);
            auto dC = make_cute_packed_stride(StrideC{}, cutlass_problem_shape.stride_C);
            auto tensor_kqpn = make_tensor(C_shared, make_layout(shape_C_orig, dC));

            auto slayoutB = layoutSB(_, _, 0);
            auto tilerB = Shape<Int<bN>, Shape<Int<bK>>>{};
            auto load_b = make_xe4_copy_conv2d<ASYNC_TENSOR_LOAD, AuxParamsB>(gtensorB, slayoutB, tilerB);

            Tensor mB_nk = load_b.get_tma_tensor(make_shape(MNKL_N, MNKL_K));
            Tensor gB_nk = local_tile(mB_nk, TileShapeMNK{}, make_coord(_, _, _), Step< X, _1, _1>{});
            
            Tensor gB_nkl = gB_nk;
            auto n_coord = idx2crd(int(wg_id_x), shape<2>(gB_nkl), compact_col_major(shape<2>(gB_nkl)));

            auto block_load_b = load_b.get_slice(wg_id_x);
            Tensor gB = gB_nk(_, _, n_coord, _);
            Tensor tBgB = block_load_b.partition_S(gB);
            Tensor tBsB = block_load_b.partition_D(sB);

            // compute the upper and lower corners based on conv padding
            auto lower_corner_whd = compute_lower_corner_whd<2>(cutlass_problem_shape);
            auto upper_corner_whd = compute_upper_corner_whd<2>(cutlass_problem_shape);
            auto lower_srt = compute_lower_srt<2>(cutlass_problem_shape);
            auto stride_srt = compute_stride_srt<2>(cutlass_problem_shape);

            auto [load_a, tma_tensor_load] = make_im2col_tma_copy<ASYNC_ROW_LOAD_IM2COL, cute::C<cm_typeA>, cute::C<dim>>(tensor_cwhn,
                                                                                                  cutlass_problem_shape.shape_A,
                                                                                                  cutlass_problem_shape.stride_A,
                                                                                                  make_layout(MmaShapeMK{}, Stride<Int<bK>, _1>{}),
                                                                                                  Layout<Shape<cute::C<LANESIZE>, _1>>{},
                                                                                                  Layout<Shape<_1, decltype(bK)>>{},
                                                                                                  MmaShapeMK{},
                                                                                                  1,
                                                                                                  shape(lower_corner_whd),
                                                                                                  shape(upper_corner_whd),
                                                                                                  cute::reverse(shape(cutlass_problem_shape.lower_padding)),
                                                                                                  cute::reverse(shape(cutlass_problem_shape.upper_padding)),
                                                                                                  cute::reverse(shape(cutlass_problem_shape.traversal_stride)),
                                                                                                  shape(lower_srt),
                                                                                                  shape(stride_srt));
            Tensor gA_mk = local_tile(tma_tensor_load, Shape<decltype(bM), decltype(bN), Shape<decltype(bK)>>{}, make_coord(_,_,_), Step<_1, X,_1>{});  //(BLK_M, BLK_K, m, k)
            Tensor gA = gA_mk(_,_,wg_id_y,_);
            ThrCopy thr_load_a = load_a.get_slice(local_id);
            Tensor tAgA = thr_load_a.partition_S(gA);
            Tensor tAsA = thr_load_a.partition_D(sA);

            auto slm_pipe_write = cutlass::xe4::make_producer_start_state<Pipeline>();
            auto k_tile_iter = cute::make_coord_iterator(shape<3>(gA_mk));
            auto k_tile_count = size<3>(gA_mk);
            for ( ; k_tile_count > 0; --k_tile_count) {
                uint32_t abar_index = slm_pipe_write.index();
                auto abar_prod = pipeline.producer_get_barrier(abar_index);

                pipeline.producer_try_wait(slm_pipe_write);
                copy(load_a.with(abar_prod), tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,abar_index));

                if(get_lane_id() == 0) {
                    pipeline.producer_commit(abar_index, slm_bytes_a + slm_bytes_b);
                    copy(load_b.with(abar_prod), tBgB(_, _, _, *k_tile_iter), tBsB(_, _, _, abar_index));
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

            auto [store_c, tma_tensor_store] = make_im2col_tma_copy<ASYNC_ROW_STORE_IM2COL, cute::C<cm_typeC>, cute::C<dim>>(tensor_kqpn,
                                                                                      cutlass_problem_shape.shape_C,
                                                                                      cutlass_problem_shape.stride_C,
                                                                                      make_layout(MmaShapeMN{}, Stride<Int<bN>, _1>{}),
                                                                                      Layout<Shape<cute::C<LANESIZE>, _1>>{},
                                                                                      Layout<Shape<_1, decltype(bN)>>{},
                                                                                      MmaShapeMN{},
                                                                                      1,
                                                                                      append<2>(Stride<_0>{}, Int<0>{}),
                                                                                      append<2>(Stride<_0>{}, Int<0>{}),
                                                                                      append<2>(Stride<_0>{}, Int<0>{}),
                                                                                      append<2>(Stride<_0>{}, Int<0>{}),
                                                                                      append<2>(Stride<_1>{}, Int<1>{}),
                                                                                      append<2>(Stride<_0>{}, Int<0>{}),
                                                                                      append<2>(Stride<_1>{}, Int<1>{}));
            Tensor gC_mn = local_tile(tma_tensor_store, Shape<decltype(bM), decltype(bN), Shape<decltype(bK)>>{}, make_coord(_,_,_), Step<_1, _1, X>{});   //BLK_M, BLK_N, m, n
            Tensor gC = gC_mn(_,_,wg_id_y,wg_id_x);
            ThrCopy thr_store_c = store_c.get_slice(local_id);
            Tensor tCsC = thr_store_c.partition_S(sC);
            Tensor tCgC = thr_store_c.partition_D(gC);

            uint32_t abar_store_cons_index = slm_pipe_store_cons.index();
            auto abar_store_cons = pipeline_store.producer_get_barrier(abar_store_cons_index);
            copy(store_c.with(abar_store_cons), tCsC, tCgC);

            ++slm_pipe_store_cons;
            pipeline_store.producer_try_wait(slm_pipe_store_cons);
        }
        else if (subgroup_id == 1){
            if (local_id == 32){
                if (kloop == 1) {
                    Pipeline::PipelineState slm_pipe_read;
                    pipeline.consumer_try_wait(slm_pipe_read);
                    cute::gemm(tiled_mma_quaternion.with(scaleOutZero, abar_cons_base), tCrC, tCrA(_,_,_,0), tCrB(_,_,_,0), accum);
                    pipeline.consumer_commit(slm_pipe_read);
                } else {
                    Pipeline::PipelineState slm_pipe_read;
                    pipeline.consumer_try_wait(slm_pipe_read);
                    cute::gemm(tiled_mma_ternary.with(scaleOutZero, abar_cons_base), tCrA(_,_,_,0), tCrB(_,_,_,0), accum);
                    pipeline.consumer_commit(slm_pipe_read);

                    for (uint32_t i = 1; i < kloop - 1; i++) {
                        ++slm_pipe_read;
                        uint32_t abar_index = slm_pipe_read.index();
                        auto abar_cons = pipeline.consumer_get_barrier(abar_index);
                        pipeline.consumer_try_wait(slm_pipe_read);
                        cute::gemm(tiled_mma_ternary.with(scaleOutOne, abar_cons), tCrA(_,_,_,abar_index), tCrB(_,_,_,abar_index), accum);
                        pipeline.consumer_commit(slm_pipe_read);
                    }
                    {
                        auto slm_pipe_store_prod = cutlass::xe4::make_producer_start_state<PipelineStore>();
                        uint32_t abar_store_prod_index = slm_pipe_store_prod.index();
                        auto abar_store_prod = pipeline_store.producer_get_barrier(abar_store_prod_index);

                        ++slm_pipe_read;
                        uint32_t abar_index = slm_pipe_read.index();

                        uint32_t phase = ((kloop - 1) / stage) & 1u;
                        pipeline.consumer_try_wait(abar_index, phase);
                        cute::gemm(tiled_mma_quaternion.with(scaleOutOne, abar_store_prod), tCrC, tCrA(_,_,_,abar_index), tCrB(_,_,_,abar_index), accum);
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
    conv2d::problem_shape_t small {{64, 8, 8, 1}, {64, 3, 3, 512}, {0, 0}, {0, 0}, {1, 1}, {1, 1}};
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
