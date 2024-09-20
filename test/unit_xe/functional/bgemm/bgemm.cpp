#include <sycl/sycl.hpp>
#include <cute/tensor.hpp>

#include "cutlass/layout/matrix.h"
#include "cutlass/detail/layout.hpp"
#include "cutlass/util/packed_stride.hpp"

#include "cute/arch/mma_xe4.hpp"
#include "cutlass/gemm/kernel/xe4_gemm_dma_warpspecialized.hpp"
#include "validation.hpp"

using namespace cute;
using namespace sycl;
using namespace cute::xe4;
using namespace cutlass::gemm::collective;

class BGEMM_ROW_ROW;
class BGEMM_COL_ROW;
class BGEMM_ROW_COL;
class BGEMM_COL_COL;
class BGEMM_ROW_ROW_RELU;
class BGEMM_COL_ROW_RELU;
class BGEMM_ROW_COL_RELU;
class BGEMM_COL_COL_RELU;

template<typename test, mem_layout layout_a, mem_layout layout_b, typename PostOp>
int run_test()
{
    queue q;
    auto dev = q.get_device();
    std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

    int mat_m = 256;
    int mat_n = 256;
    int mat_k = 512;
    int mat_l = 1;
    constexpr uint32_t wg_m = 128;
    constexpr uint32_t wg_n = 128;
    constexpr uint32_t wg_k = 128;
    constexpr uint32_t stage = 3;

    assert(((mat_k + wg_k - 1) / wg_k) > 1);

    uint32_t sizeA = mat_m * mat_k;
    uint32_t sizeB = mat_n * mat_k;
    uint32_t sizeC = mat_m * mat_n;

    using dtypeA = bf16;
    using dtypeB = bf16;
    using dtypeAcc = float;
    using dtypeC = fp16;

    static constexpr bool is_row_major_a = (layout_a == mem_layout::row_major);
    static constexpr bool is_row_major_b = (layout_b == mem_layout::row_major);

    using EpilogueOp = std::conditional_t<std::is_same_v<PostOp, ReLu>,
            DMAPostOPReLu<dtypeC, dtypeAcc, 16, 32>, DMAPostOPConvert<dtypeC, dtypeAcc, 16, 32>>;

    using LayoutA = std::conditional_t<is_row_major_a, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>;
    using LayoutB = std::conditional_t<is_row_major_b, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>;
    using LayoutC = cutlass::layout::RowMajor;

    auto A_s = malloc_shared<dtypeA>(sizeA, q);
    std::generate_n(A_s, sizeA, [=]() { return static_cast<float>(rand()) / static_cast<float>(RAND_MAX); });

    auto B_s = malloc_shared<dtypeB>(sizeB, q);
    std::generate_n(B_s, sizeB, [=]() { return static_cast<float>(rand()) / static_cast<float>(RAND_MAX); });

    auto C_s = malloc_shared<dtypeC>(sizeC, q);
    std::fill_n(C_s, sizeC, dtypeC(0));

    range<3> local_range(1, 20, 32);
    uint32_t group_range_m = (mat_m + wg_m - 1) / wg_m;
    uint32_t group_range_n = (mat_n + wg_n - 1) / wg_n;
    range<3> group_range(1, group_range_m, group_range_n);
    nd_range<3> Range(group_range * local_range, local_range);

    std::cout << "ProblemShape: (" << mat_m << ", " << mat_n << ", " << mat_k << ")\n";
    std::cout << "TileShape: (" << wg_m << ", " << wg_n << ", " << wg_k << ")\n";
    std::cout << "isRowMajorA=" << is_row_major_a << " isRowMajorB=" << is_row_major_b << "\n";
    std::cout << "Group range: {" << 1 << ", " << group_range_m << ", " << group_range_n << "} \n";

    using StrideA = cutlass::detail::TagToStrideA_t<LayoutA>;
    using StrideB = cutlass::detail::TagToStrideB_t<LayoutB>;
    using StrideC = cutlass::detail::TagToStrideC_t<LayoutC>;

    using TileShape = Shape<Int<wg_m>, Int<wg_n>, Int<wg_k>>;
    using TiledMma = decltype(cute::make_tiled_mma(AMMA::ss_op_selector<AMMA::OpType::NoneCluster, dtypeA, dtypeB, dtypeAcc, TileShape, is_row_major_a, is_row_major_b>()));

    using SmemLayoutAtomA = decltype(upcast<sizeof(dtypeA)>(make_layout(Shape<_32,_32>{}, GenRowMajor{})));
    using SmemLayoutAtomB = decltype(upcast<sizeof(dtypeB)>(make_layout(Shape<_32,_32>{}, std::conditional_t<is_row_major_b, GenColMajor, GenRowMajor>{})));

    using SmemLayoutAtomC = Layout<Shape<Int<wg_m>, Int<wg_n>>, Stride<Int<wg_n>, _1>>;

    using CollectiveMainloop = CollectiveMma<
        MainloopXe4DmaGmma<stage>,                                                              // MainloopXe4DmaGmma
        TileShape,                                                                              // TileShape
        dtypeA,                                                                                 // ElementA
        StrideA,                                                                                // StrideA
        dtypeB,                                                                                 // ElementB
        StrideB,                                                                                // StrideB
        TiledMma,                                                                               // TiledMma
        ASYNC_TENSOR_LOAD,                                                                      // GmemTiledCopyA
        SmemLayoutAtomA,                                                                        // SmemLayoutAtomA
        void,                                                                                   // SmemCopyAtomA
        void,                                                                                   // TransformA
        ASYNC_TENSOR_LOAD,                                                                      // GmemTiledCopyB
        SmemLayoutAtomB,                                                                        // SmemLayoutAtomB
        void,                                                                                   // SmemCopyAtomB
        void                                                                                    // TransformB
    >;

    using CollectiveEpilogue = cutlass::epilogue::collective::DefaultEpilogue<
        StrideC,
        StrideC,
        SmemLayoutAtomC,
        SmemLayoutAtomC,
        decltype(take<0, 2>(TileShape{})),
        EpilogueOp,
        cutlass::gemm::EpilogueDefault
    >;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int,int,int,int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        void
    >;

    q.parallel_for<test>(Range, [=](nd_item<3> item) {
        auto args = typename GemmKernel::Arguments {
            item,
            make_shape(mat_m, mat_n, mat_k, mat_l),
            {
                A_s, cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(mat_m, mat_k, mat_l)),
                B_s, cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(mat_n, mat_k, mat_l)),
            },
            {
                C_s, cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(mat_m, mat_n, mat_l)),
            }
        };

        GemmKernel kernel;
        auto params = kernel.to_underlying_arguments(args, nullptr);
        kernel(params);
     }).wait();

    uint32_t err_cnt = validate_gemm_result(A_s, B_s, C_s, mat_m, mat_n, mat_k, layout_a, layout_b, PostOp{});
    if (err_cnt > 0) {
        std::cout << "Test Failed!" << std::endl;
        return -1;
    }

    std::cout << "Test Pass!" << std::endl;
    return 0;
}

int main()
{

#if defined(TEST_ROW_ROW)
    run_test<BGEMM_ROW_ROW, mem_layout::row_major, mem_layout::row_major, DoNothing>();
#elif defined(TEST_COL_ROW)
    run_test<BGEMM_COL_ROW, mem_layout::col_major, mem_layout::row_major, DoNothing>();
#elif defined(TEST_ROW_COL)
    run_test<BGEMM_ROW_COL, mem_layout::row_major, mem_layout::col_major, DoNothing>();
#elif defined(TEST_COL_COL)
    run_test<BGEMM_COL_COL, mem_layout::col_major, mem_layout::col_major, DoNothing>();
#elif defined(TEST_ROW_ROW_RELU)
    run_test<BGEMM_ROW_ROW_RELU, mem_layout::row_major, mem_layout::row_major, ReLu>();
#elif defined(TEST_COL_ROW_RELU)
    run_test<BGEMM_COL_ROW_RELU, mem_layout::col_major, mem_layout::row_major, ReLu>();
#elif defined(TEST_ROW_COL_RELU)
    run_test<BGEMM_ROW_COL_RELU, mem_layout::row_major, mem_layout::col_major, ReLu>();
#elif defined(TEST_COL_COL_RELU)
    run_test<BGEMM_COL_COL_RELU, mem_layout::col_major, mem_layout::col_major, ReLu>();
#endif

    return 0;
}
