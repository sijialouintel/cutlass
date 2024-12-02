#include <CL/sycl.hpp>
#include "validation.hpp"
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/arch/mma_xe4.hpp"
#include "cutlass/conv/collective/xe4_implicit_gemm_gmma_ss_warpspecialized.hpp"
#include "cutlass/conv/kernel/xe4_implicit_gemm_dma_warpspecialized.hpp"

using namespace sycl;
using namespace cutlass::xe4;
using namespace cute;
using namespace cute::detail;
using namespace cute::xe4;
using namespace cutlass::conv::collective;
using namespace cutlass::epilogue::collective;

class CONV2D_SMALL;
class CONV2D_LARGE;
class CONV2D_SMALL_WITH_PAD_WITH_STRIDE;
class CONV2D_LARGE_WITH_PAD_WITH_STRIDE;
class CONV2D_OTHER_WITH_PAD_WITH_STRIDE;
class CONV2D_ASYNMMETRIC_PAD_ASYNMMETRIC_STRIDE;
class CONV2D_LARGE_WITH_PAD_WITH_STRIDE_WITH_DILATION;
class CONV2D_OTHER_WITH_PAD_WITH_STRIDE_WITH_DILATION;
class CONV2D_ASYNMMETRIC_PAD_ASYNMMETRIC_STRIDE_WITH_DILATION;

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

    using ElementAct = bf16;
    using ElementFlt = bf16;
    using ElementAcc = float;
    using ElementOut = bf16;

    constexpr uint32_t Stages = 3;
    static constexpr auto tnspA = xe4::GMMA::Major::K;
    static constexpr auto tnspB = xe4::GMMA::Major::K;

    auto bM = Int<64>{};
    auto bN = Int<256>{};
    auto bK = Int<128>{};
    static_assert(bM % LANESIZE == 0);
    using TileShape = Shape<Int<bM>, Int<bN>, Shape<Int<bK>>>;
    using ClusterShapeMNK = Shape<_1, _1, _1>;
    using MmaTiler = Shape<Int<bM>, Int<bN>, Int<bK>>;
    using TiledMma = decltype(cute::make_tiled_mma(xe4::GMMA::ss_op_selector<xe4::GMMA::OpType::NoneCluster, ElementAct, ElementFlt, ElementAcc, ElementOut, MmaTiler, tnspA, tnspB>()));
    using SmemLayoutAtomA = decltype(make_layout(Shape<_32, Int<32 / sizeof(ElementAct)>>{}, std::conditional_t<tnspA == xe4::GMMA::Major::K, GenRowMajor, GenColMajor>{}));
    using SmemLayoutAtomB = decltype(upcast<sizeof(ElementFlt)>(make_layout(Shape<_32, _32>{}, std::conditional_t<tnspB == xe4::GMMA::Major::K, GenRowMajor, GenColMajor>{})));
    using SmemLayoutC = decltype(make_layout(make_shape(bM, bN), make_stride(bN, Int<1>{})));

    uint32_t sizeA = C * W * H * N;
    uint32_t sizeB = C * S * R * K;
    uint32_t sizeC = Out_C * Out_W * Out_H * Out_N;

    auto A_shared = malloc_shared<ElementAct>(sizeA, q);
    std::generate_n(A_shared, sizeA, [=]() { return static_cast<float>(rand()) / static_cast<float>(RAND_MAX); });

    auto B_shared = malloc_shared<ElementFlt>(sizeB, q);
    std::generate_n(B_shared, sizeB, [=]() { return static_cast<float>(rand()) / static_cast<float>(RAND_MAX); });

    auto C_shared = malloc_shared<ElementOut>(sizeC, q);
    std::fill_n(C_shared, sizeC, ElementOut(0));

    range<3> local_range(1, 2, 64);
    uint32_t mat_m = Out_W * Out_H * Out_N;
    uint32_t mat_n = K;
    uint32_t group_range_m = (mat_m + bM - 1) / bM;
    uint32_t group_range_n = (mat_n + bN - 1) / bN;
    range<3> group_range(1, group_range_m, group_range_n);
    std::cout << "Group range: {" << 1 << ", " << group_range_m << ", " << group_range_n << "} \n";
    nd_range<3> Range(group_range * local_range, local_range);

    using CollectiveMainloop = CollectiveConv<
        cute::C<cutlass::conv::Operator::kFprop>,
        Stages,
        2,
        ClusterShapeMNK,
        KernelImplicitTmaWarpSpecializedXe4,
        1,
        TileShape,
        ElementAct,
        ElementFlt,
        TiledMma,
        SmemLayoutAtomA,
        SmemLayoutAtomB>;
    using CollectiveEpilogue = EpilogueConv<
        cutlass::conv::Operator::kFprop,
        CollectiveMainloop::DispatchPolicy::NumSpatialDimensions,
        SmemLayoutC,
        TileShape,
        ElementOut>;
    using ProblemShape = cutlass::conv::ConvProblemShape<
        cutlass::conv::Operator::kFprop,
        CollectiveMainloop::DispatchPolicy::NumSpatialDimensions>;
    using ConvKernel = cutlass::conv::kernel::Xe4ConvUniversal<
        ProblemShape,
        CollectiveMainloop,
        CollectiveEpilogue,
        void>;

    q.parallel_for<test>(Range, [=](nd_item<3> item) {
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

        auto args = typename ConvKernel::Arguments {
            cutlass_problem_shape,
            {A_shared, B_shared},
            {C_shared}
        };

        ConvKernel kernel;
        auto params = kernel.to_underlying_arguments(args, nullptr);
        kernel(params, item);
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
