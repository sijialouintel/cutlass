#include <sstream>
#include <sycl/sycl.hpp>
#include <cute/tensor.hpp>

#include "cutlass/layout/matrix.h"
#include "cutlass/detail/layout.hpp"
#include "cutlass/util/packed_stride.hpp"

#include "cute/arch/mma_xe4.hpp"
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

struct BGEMM_ROW_ROW
{
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    static constexpr uint32_t cluster_size_x = 2;
    static constexpr uint32_t cluster_size_y = 2;
};


struct BGEMM_COL_ROW
{
    static constexpr mem_layout layout_a = mem_layout::col_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    static constexpr uint32_t cluster_size_x = 2;
    static constexpr uint32_t cluster_size_y = 2;
};

struct BGEMM_ROW_COL
{
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::col_major;
    static constexpr uint32_t cluster_size_x = 2;
    static constexpr uint32_t cluster_size_y = 2;
};

struct relu_op_t
{
    template<typename dtype_acc>
    void run(std::vector<dtype_acc>& gold_acc){
        for(size_t i = 0; i < gold_acc.size(); i++){
            gold_acc[i] = gold_acc[i] > 0 ? gold_acc[i] : 0;
        }
    }
};

template<typename test>
void run_test()
{
    queue q;
    auto dev = q.get_device();
    std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

    int mat_m = 1024;
    int mat_n = 1024;
    int mat_k = 1024;
    int mat_l = 1;
    constexpr uint32_t wg_m = 256;
    constexpr uint32_t wg_n = 512;
    constexpr uint32_t wg_k = 128;
    constexpr uint32_t stage = 3;

    constexpr mem_layout layout_a = test::layout_a;
    constexpr mem_layout layout_b = test::layout_b;
    constexpr uint32_t cluster_size_x = test::cluster_size_x;
    constexpr uint32_t cluster_size_y = test::cluster_size_y;
    constexpr uint32_t cluster_size = cluster_size_x * cluster_size_y;

    std::stringstream ss;
    ss << cluster_size_x << "x" << cluster_size_y << "x1";
    std::string cluster_size_config = ss.str();
    setenv("XE4_CLUSTER_SIZE", cluster_size_config.c_str(), 1);
    std::cout << "cluster size is: " << cluster_size_config << std::endl;

    uint32_t sizeA = mat_m * mat_k;
    uint32_t sizeB = mat_n * mat_k;
    uint32_t sizeC = mat_m * mat_n;

    using dtypeA = bf16;
    using dtypeB = bf16;
    using dtypeAcc = float;
    using dtypeC = bf16;

    static constexpr auto tnspA = (layout_a == mem_layout::row_major) ? xe4::GMMA::Major::K : xe4::GMMA::Major::MN;
    static constexpr auto tnspB = (layout_b == mem_layout::row_major) ? xe4::GMMA::Major::MN : xe4::GMMA::Major::K;

    auto A_s = malloc_shared<dtypeA>(sizeA, q);
    std::generate_n(A_s, sizeA, [=]() { return static_cast<float>(rand()) / static_cast<float>(RAND_MAX); });

    auto B_s = malloc_shared<dtypeB>(sizeB, q);
    std::generate_n(B_s, sizeB, [=]() { return static_cast<float>(rand()) / static_cast<float>(RAND_MAX); });

    auto C_s = malloc_shared<dtypeC>(sizeC, q);
    std::fill_n(C_s, sizeC, dtypeC(0));

    constexpr uint32_t SubGroupSize = 32;
    constexpr uint32_t NumControlSubGroup = 4;
    constexpr uint32_t NumPostOpSubGroup = 16;
    range<3> local_range(1, NumControlSubGroup + NumPostOpSubGroup, SubGroupSize);
    uint32_t group_range_m = round_up(ceil_div(mat_m, wg_m), cluster_size_y);
    uint32_t group_range_n = round_up(ceil_div(mat_n, wg_n), cluster_size_x);
    range<3> group_range(1, group_range_m, group_range_n);
    std::cout << "Group range: {" << 1 << ", " << group_range_m << ", " << group_range_n << "} \n";
    nd_range<3> Range(group_range * local_range, local_range);

    using LayoutA = std::conditional_t<tnspA == xe4::GMMA::Major::K, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>;
    using StrideA = cutlass::detail::TagToStrideA_t<LayoutA>;
    using LayoutB = std::conditional_t<tnspB == xe4::GMMA::Major::MN, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>;
    using StrideB = cutlass::detail::TagToStrideB_t<LayoutB>;
    using LayoutC = cutlass::layout::RowMajor;
    using StrideC = cutlass::detail::TagToStrideC_t<LayoutC>;

    using ClusterShape = Shape<Int<cluster_size_y>,Int<cluster_size_x>,_1>;
    using TileShape = Shape<Int<wg_m>, Int<wg_n>, Int<wg_k>>;
    using TiledMma = decltype(cute::make_tiled_mma(xe4::GMMA::ss_op_selector<xe4::GMMA::OpType::Cluster, dtypeA, dtypeB, dtypeAcc, dtypeC, TileShape, tnspA, tnspB>()));

    using SmemLayoutAtomA = decltype(make_layout(Shape<_32,Int<32/sizeof(dtypeA)>>{}, std::conditional_t<tnspA == xe4::GMMA::Major::K, GenRowMajor, GenColMajor>{}));
    using SmemLayoutAtomB = decltype(upcast<sizeof(dtypeB)>(make_layout(Shape<_32,_32>{}, std::conditional_t<tnspB == xe4::GMMA::Major::K, GenRowMajor, GenColMajor>{})));

    using SmemLayoutAtomC = Layout<Shape<Int<wg_m>, Int<wg_n>>, Stride<Int<wg_n>, _1>>;

    using CollectiveMainloop = CollectiveMma<
        MainloopXe4DmaGmmaWarpSpecialized<stage, ClusterShape>,                                 // DispatchPolicy
        TileShape,                                                                              // TileShape
        dtypeA,                                                                                 // ElementA
        StrideA,                                                                                // StrideA
        dtypeB,                                                                                 // ElementB
        StrideB,                                                                                // StrideB
        TiledMma,                                                                               // TiledMma
        ASYNC_TENSOR_LOAD_MULTICAST,                                                            // GmemTiledCopyA
        SmemLayoutAtomA,                                                                        // SmemLayoutAtomA
        void,                                                                                   // SmemCopyAtomA
        void,                                                                                   // TransformA
        ASYNC_TENSOR_LOAD_MULTICAST,                                                            // GmemTiledCopyB
        SmemLayoutAtomB,                                                                        // SmemLayoutAtomB
        void,                                                                                   // SmemCopyAtomB
        void                                                                                    // TransformB
    >;

    static constexpr int FragmentSize = 2;
    using FusionOp = fusion::LinCombEltAct<thread::ReLu, dtypeC, dtypeC, void>;  // LinCombEltAct<ElementOutput, ElementCompute, ElementSource>
    using FusionCallbacks = fusion::FusionCallbacks<
        Sm90TmaWarpSpecialized<1, 1, FragmentSize, false, false>,
        FusionOp, Shape<Int<wg_m>, Int<wg_n>, _1>, Shape<_2,_1>
    >;

    using CollectiveEpilogue = CollectiveEpilogue<
        Xe4DmaWarpSpecialized<FragmentSize, NumPostOpSubGroup, SubGroupSize>,
        dtypeC, StrideC, SmemLayoutAtomC, Shape<Int<wg_m>, Int<wg_n>>, FusionCallbacks
    >;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int,int,int,int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        void
    >;

    q.parallel_for<test>(Range, [=](nd_item<3> item) {
        auto args = typename GemmKernel::Arguments {
            {SubGroupSize, NumControlSubGroup, NumPostOpSubGroup},
            make_shape(mat_m, mat_n, mat_k, mat_l),
            {
                A_s, cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(mat_m, mat_k, mat_l)),
                B_s, cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(mat_n, mat_k, mat_l)),
            },
            {
                typename FusionCallbacks::Arguments{},
                C_s, cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(mat_m, mat_n, mat_l)),
            }
        };

        GemmKernel kernel;
        auto params = kernel.to_underlying_arguments(args, nullptr);
        kernel(params, item);
     }).wait();

    uint32_t err_cnt = validate_gemm_result(A_s, B_s, C_s, mat_m, mat_n, mat_k, layout_a, layout_b, relu_op_t{});

    if (err_cnt > 0) {
        throw std::runtime_error("Test Failed!");
    } else {
        std::cout << "Test Pass!" << std::endl;
    }
}

int main()
{
    run_test<BGEMM_ROW_ROW>();
    run_test<BGEMM_COL_ROW>();
    run_test<BGEMM_ROW_COL>();
    return 0;
}
