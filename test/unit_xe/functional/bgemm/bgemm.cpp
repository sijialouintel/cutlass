#include <sycl/sycl.hpp>
#include <cute/tensor.hpp>

#include "collective_mma.hpp"
#include "xe4_copy_async.hpp"
#include "validation.hpp"

using namespace cute;
using namespace sycl;
using namespace cute::xe4;

class BGEMM;

int main()
{
    queue q;
    auto dev = q.get_device();
    std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";
    auto ctxt = q.get_context();

    uint32_t mat_m = 1024;
    uint32_t mat_n = 1024;
    uint32_t mat_k = 1024;
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

    auto A_s = malloc_shared<dtypeA>(sizeA, q);
    std::generate_n(A_s, sizeA, [=]() { return static_cast<float>(rand()) / static_cast<float>(RAND_MAX); });

    auto B_s = malloc_shared<dtypeB>(sizeB, q);
    std::generate_n(B_s, sizeB, [=]() { return static_cast<float>(rand()) / static_cast<float>(RAND_MAX); });

    auto C_s = malloc_shared<dtypeC>(sizeC, q);
    std::fill_n(C_s, sizeC, dtypeC(0));

    range<3> local_range(1, 1, 64);
    uint32_t group_range_m = (mat_m + wg_m - 1) / wg_m;
    uint32_t group_range_n = (mat_n + wg_n - 1) / wg_n;
    range<3> group_range(1, group_range_m, group_range_n);
    std::cout << "Group range: {" << 1 << ", " << group_range_m << ", " << group_range_n << "} \n";
    nd_range<3> Range(group_range * local_range, local_range);

    using mat_desc_t = uint64_t;
    using abar_ptr_t = uint64_t*;
    using tdesc_ptr_t = uint64_t*;

    using TileShape = Shape<Int<wg_m>, Int<wg_n>, Int<wg_k>>;

    using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
        stage,                                                                                  // Stages
        TileShape,                                                                              // TileShape
        dtypeA,                                                                                 // ElementA
        cute::Stride<int64_t, cute::Int<1>>,                                                    // StrideA
        dtypeB,                                                                                 // ElementB
        cute::Stride<int64_t, cute::Int<1>>,                                                    // StrideB
        dtypeC,                                                                                 // ElementC
        dtypeAcc,                                                                               // ElementAcc
        cute::Stride<int64_t, cute::Int<1>>,                                                    // StrideC
        ASYNC_TENSOR_LOAD,                                                                      // GmemTiledCopyA
        Layout<Shape<Int<wg_m>,Int<wg_k>,Int<stage>>, Stride<Int<wg_k>,_1,Int<wg_m*wg_k>>>,     // SmemLayoutAtomA
        ASYNC_TENSOR_LOAD,                                                                      // GmemTiledCopyB
        Layout<Shape<Int<wg_k>,Int<wg_n>,Int<stage>>, Stride<Int<wg_n>,_1,Int<wg_k*wg_n>>>,     // SmemLayoutAtomB
        ASYNC_TENSOR_STORE,                                                                     // GmemTiledCopyC
        Layout<Shape<Int<wg_m>,Int<wg_n>>, Stride<Int<wg_n>,_1>>,                               // SmemLayoutAtomC
        tdesc_ptr_t,                                                                            // TensorDescPtr
        abar_ptr_t,                                                                             // AbarrierPtr
        mat_desc_t                                                                              // MatrixDesc
    >;

    using Pipeline = typename CollectiveMainloop::MainloopPipeline;
    using PipelineState = typename CollectiveMainloop::PipelineState;

    q.parallel_for<class BGEMM>(Range, [=](nd_item<3> item) {
        auto problem_shape = make_shape(mat_m, mat_n, mat_k);
        auto args = CollectiveMainloop::Arguments {
            A_s, make_stride(mat_k, _1{}),
            B_s, make_stride(mat_n, _1{}),
            C_s, make_stride(mat_n, _1{}),
            item.get_group(),
        };

        auto mainloop = CollectiveMainloop{};
        auto params = mainloop.to_underlying_arguments(problem_shape, args);
        auto load_inputs = mainloop.load_init(problem_shape, params);

        Pipeline pipeline(item);
        uint32_t local_id = item.get_local_id(2);
        uint32_t k_tile_count = (mat_k + wg_k -1) / wg_k;
        auto blk_coord = cute::make_tuple(item.get_group(1), item.get_group(2));
        auto slm_pipe_write = cutlass::xe4::make_producer_start_state<Pipeline>();
        mainloop.load(params, pipeline, slm_pipe_write, load_inputs, blk_coord, k_tile_count, local_id);

        PipelineState slm_pipe_read;
        mainloop.mma(params, pipeline, slm_pipe_read, k_tile_count, local_id);
     }).wait();

    uint32_t err_cnt = validate_gemm_result(A_s, B_s, C_s, mat_m, mat_n, mat_k);
    if (err_cnt > 0) {
        std::cout << "Test Failed!" << std::endl;
        return -1;
    }

    std::cout << "Test Pass!" << std::endl;
    return 0;
}
