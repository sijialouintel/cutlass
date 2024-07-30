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

template <class T, uint32_t bM, uint32_t bN>
void transpose_block(T* src, T* dst, int m, int n) {
    for (int i = 0; i < m; i+=bM) {
        for (int j = 0; j < n; j+=bN) {
            for (int ii = 0; ii < bM; ++ii) {
                for (int jj = 0; jj < bN; ++jj) {
                    dst[(i+ii)*n+j+jj] = src[(i+jj)*n+j+ii];
                }
            }
        }
    }
}

int main()
{
    queue q;
    auto dev = q.get_device();
    std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";
    auto ctxt = q.get_context();

    int mat_m = 1024;
    int mat_n = 1024;
    int mat_k = 2048;
    int mat_l = 1;
    constexpr uint32_t wg_m = 128;
    constexpr uint32_t wg_n = 128;
    constexpr uint32_t wg_k = 128;
    constexpr uint32_t stage = 4;

    uint32_t sizeA = mat_m * mat_k;
    uint32_t sizeB = mat_n * mat_k;
    uint32_t sizeC = mat_m * mat_n;

    using dtypeA = bf16;
    using dtypeB = bf16;
    using dtypeAcc = float;
    using dtypeC = float;

    static constexpr bool transposeA = false;
    static constexpr bool transposeB = false;

    using LayoutA = std::conditional_t<transposeA, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>;
    using LayoutB = std::conditional_t<transposeB, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>;
    using LayoutC = cutlass::layout::RowMajor;

    std::vector<dtypeA> A_h(sizeA);
    std::generate_n(A_h.data(), sizeA, [=]() { return static_cast<float>(rand()) / static_cast<float>(RAND_MAX); });

    auto A_s = malloc_shared<dtypeA>(sizeA, q);
    if constexpr (transposeA) {
        transpose_block<dtypeA, wg_k, wg_m>(A_h.data(), A_s, mat_k, mat_m);
    } else {
        std::copy_n(A_h.data(), sizeA, A_s);
    }

    std::vector<dtypeB> B_h(sizeB);
    std::generate_n(B_h.data(), sizeB, [=]() { return static_cast<float>(rand()) / static_cast<float>(RAND_MAX); });

    auto B_s = malloc_shared<dtypeB>(sizeB, q);
    if constexpr (transposeB) {
        transpose_block<dtypeB, wg_n, wg_k>(B_h.data(), B_s, mat_n, mat_k);
    } else {
        std::copy_n(B_h.data(), sizeB, B_s);
    }

    auto C_s = malloc_shared<dtypeC>(sizeC, q);
    std::fill_n(C_s, sizeC, dtypeC(0));

    range<3> local_range(1, 1, 64);
    uint32_t group_range_m = (mat_m + wg_m - 1) / wg_m;
    uint32_t group_range_n = (mat_n + wg_n - 1) / wg_n;
    range<3> group_range(1, group_range_m, group_range_n);
    std::cout << "Group range: {" << 1 << ", " << group_range_m << ", " << group_range_n << "} \n";
    nd_range<3> Range(group_range * local_range, local_range);

    using StrideA = cutlass::detail::TagToStrideA_t<LayoutA>;
    using StrideB = cutlass::detail::TagToStrideB_t<LayoutB>;
    using StrideC = cutlass::detail::TagToStrideC_t<LayoutC>;

    using TileShape = Shape<Int<wg_m>, Int<wg_n>, Int<wg_k>>;
    using MMA_Op = XE4_ASYNC_GMMA<dtypeAcc, void, dtypeA, dtypeB, TileShape,
        CoreMatrixSize<cm_size_t::cm_32x32B, cm_size_t::cm_16x32B, cm_size_t::cm_32x32B>, uint64_t, uint64_t*>;

    using CollectiveMainloop = CollectiveMma<
        MainloopXe4DmaGmma<stage>,                                                              // MainloopXe4DmaGmma
        TileShape,                                                                              // TileShape
        dtypeA,                                                                                 // ElementA
        StrideA,                                                                                // StrideA
        dtypeB,                                                                                 // ElementB
        StrideB,                                                                                // StrideB
        decltype(cute::make_tiled_mma(MMA_Op{})),                                               // TiledMma
        ASYNC_TENSOR_LOAD,                                                                      // GmemTiledCopyA
        Layout<Shape<Int<wg_m>,Int<wg_k>,Int<stage>>, Stride<Int<wg_k>,_1,Int<wg_m*wg_k>>>,     // SmemLayoutAtomA
        void,                                                                                   // SmemCopyAtomA
        void,                                                                                   // TransformA
        ASYNC_TENSOR_LOAD,                                                                      // GmemTiledCopyB
        Layout<Shape<Int<wg_n>,Int<wg_k>,Int<stage>>, Stride<_1,Int<wg_n>,Int<wg_k*wg_n>>>,     // SmemLayoutAtomB
        void,                                                                                   // SmemCopyAtomB
        void                                                                                    // TransformB
    >;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int,int,int,int>,
        CollectiveMainloop,
        void,
        void
    >;

    q.parallel_for<class BGEMM>(Range, [=](nd_item<3> item) {
        auto problem_shape = make_shape(mat_m, mat_n, mat_k, mat_l);
        auto args = GemmKernel::Arguments {
            item,
            problem_shape,
            {
                A_s, cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(mat_m, mat_k, mat_l)),
                B_s, cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(mat_n, mat_k, mat_l)),
                C_s, cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(mat_m, mat_n, mat_l)),
                item.get_group(),
            }
        };

        GemmKernel kernel;
        auto params = kernel.to_underlying_arguments(args, nullptr);
        kernel(params);
     }).wait();

    mem_layout layout_a = transposeA ? mem_layout::col_major : mem_layout::row_major;
    mem_layout layout_b = transposeB ? mem_layout::col_major : mem_layout::row_major;
    uint32_t err_cnt = validate_gemm_result(A_h.data(), B_h.data(), C_s, mat_m, mat_n, mat_k, layout_a, layout_b);
    if (err_cnt > 0) {
        std::cout << "Test Failed!" << std::endl;
        return -1;
    }

    std::cout << "Test Pass!" << std::endl;
    return 0;
}
