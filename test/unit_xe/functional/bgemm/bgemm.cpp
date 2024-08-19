#include <sycl/sycl.hpp>
#include <cute/tensor.hpp>

#include "cutlass/layout/matrix.h"
#include "cutlass/detail/layout.hpp"
#include "cutlass/util/packed_stride.hpp"

#include "xe4_gemm.hpp"
#include "validation.hpp"

using namespace cute;
using namespace sycl;
using namespace cute::xe4;
using namespace cutlass::gemm::collective;

class BGEMM;

int main()
{
    queue q;
    auto dev = q.get_device();
    std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";
    auto ctxt = q.get_context();

    int mat_m = 256;
    int mat_n = 512;
    int mat_k = 512;
    int mat_l = 1;
    constexpr uint32_t wg_m = 256;
    constexpr uint32_t wg_n = 512;
    constexpr uint32_t wg_k = 128;
    constexpr uint32_t stage = 4;

    uint32_t sizeA = mat_m * mat_k;
    uint32_t sizeB = mat_n * mat_k;
    uint32_t sizeC = mat_m * mat_n;

    using dtypeA = bf16;
    using dtypeB = bf16;
    using dtypeAcc = float;
    using dtypeC = float;

    static constexpr mem_layout layout_a = mem_layout::col_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;

    static constexpr bool is_row_major_a = (layout_a == mem_layout::row_major);
    static constexpr bool is_row_major_b = (layout_b == mem_layout::row_major);

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
    std::cout << "Group range: {" << 1 << ", " << group_range_m << ", " << group_range_n << "} \n";
    nd_range<3> Range(group_range * local_range, local_range);

    using StrideA = cutlass::detail::TagToStrideA_t<LayoutA>;
    using StrideB = cutlass::detail::TagToStrideB_t<LayoutB>;
    using StrideC = cutlass::detail::TagToStrideC_t<LayoutC>;

    using TileShape = Shape<Int<wg_m>, Int<wg_n>, Int<wg_k>>;
    using MMA_Op = XE4_ASYNC_GMMA<dtypeAcc, void, dtypeA, dtypeB, TileShape, is_row_major_a, is_row_major_b, uint32_t, uint64_t*>;

    using SmemLayoutAtomA = std::conditional_t<is_row_major_a,
        Layout<Shape<Int<wg_m>,Int<wg_k>,Int<stage>>, Stride<Int<wg_k>,_1,Int<wg_m*wg_k>>>,
        Layout<Shape<Int<wg_m>,Int<wg_k>,Int<stage>>, Stride<_1,Int<wg_m>,Int<wg_m*wg_k>>>
    >;

    using SmemLayoutAtomB = std::conditional_t<is_row_major_b,
        Layout<Shape<Int<wg_n>,Int<wg_k>,Int<stage>>, Stride<_1,Int<wg_n>,Int<wg_k*wg_n>>>,
        Layout<Shape<Int<wg_n>,Int<wg_k>,Int<stage>>, Stride<Int<wg_k>,_1,Int<wg_k*wg_n>>>
    >;

    using CollectiveMainloop = CollectiveMma<
        MainloopXe4DmaGmma<stage>,                                                              // MainloopXe4DmaGmma
        TileShape,                                                                              // TileShape
        dtypeA,                                                                                 // ElementA
        StrideA,                                                                                // StrideA
        dtypeB,                                                                                 // ElementB
        StrideB,                                                                                // StrideB
        decltype(cute::make_tiled_mma(MMA_Op{})),                                               // TiledMma
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
        1,
        StrideC,
        DummyConverter<decltype(take<0, 2>(TileShape{})), dtypeC, dtypeAcc>,
        cutlass::gemm::EpilogueDefault
    >;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int,int,int,int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        void
    >;

    q.parallel_for<class BGEMM>(Range, [=](nd_item<3> item) {
        uint32_t wg_id = item.get_group().get_group_linear_id();
        auto problem_shape = make_shape(mat_m, mat_n, mat_k, mat_l);

        auto args = GemmKernel::Arguments {
            item,
            problem_shape,
            {
                wg_id,
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

    uint32_t err_cnt = validate_gemm_result(A_s, B_s, C_s, mat_m, mat_n, mat_k, layout_a, layout_b);
    if (err_cnt > 0) {
        std::cout << "Test Failed!" << std::endl;
        return -1;
    }

    std::cout << "Test Pass!" << std::endl;
    return 0;
}
