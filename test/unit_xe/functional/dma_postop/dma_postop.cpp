#define __RUN_ON_CPU__ 0
#include <cute/tensor.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/util/packed_stride.hpp>

#include <CL/sycl.hpp>

#include "inline_pisa.hpp"
#include "cutlass/epilogue/thread/xe4_conversion_op.hpp"
#include "cutlass/epilogue/thread/xe4_relu_op.hpp"
#include "cute/atom/copy_traits_xe4_dma.hpp"
#include "cutlass/epilogue/collective/xe4_epilogue_dma_warpspecialized.hpp"

using namespace sycl;
using namespace cute;
using namespace cute::detail;
using namespace cute::xe4;

class DMA
{
};

int main()
{
    using dtype_src = uint32_t;
    using dtype_dst = uint8_t;
    queue q;
    auto dev = q.get_device();
    std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";
    auto ctxt = q.get_context();

    constexpr int32_t gmemSizeX = 128;
    constexpr int32_t gmemSizeY = 256;
    constexpr int32_t gmemSize = gmemSizeX * gmemSizeY;

    auto *A_d = malloc_device<dtype_src>(gmemSize, q);
    auto *B_d = malloc_device<dtype_dst>(gmemSize, q);

    std::vector<dtype_src> A_h(gmemSize);
    std::vector<dtype_dst> B_h(gmemSize);

    for (auto i = 0; i != gmemSize; i++)
    {
        A_h[i] = i;
        B_h[i] = 0;
    }

    q.memcpy(A_d, A_h.data(), gmemSize * sizeof(dtype_src)).wait();
    q.memcpy(B_d, B_h.data(), gmemSize * sizeof(dtype_dst)).wait();

    constexpr int32_t boxSizeX = 64;
    constexpr int32_t boxSizeY = 32;
    constexpr int32_t boxSize = boxSizeX * boxSizeY;

    constexpr uint32_t group_range_Y = (gmemSizeY + boxSizeY - 1) / boxSizeY;
    constexpr uint32_t group_range_X = (gmemSizeX + boxSizeX - 1) / boxSizeX;
    range<3> local_range(1, 20, 32);
    range<3> group_range(1, group_range_Y, group_range_X);
    nd_range<3> Range(group_range * local_range, local_range);

    q.submit([&](handler &cgh) {
        cgh.parallel_for<class DMA>(Range, [=](nd_item<3> item) {
            using TileShape = Shape<Int<boxSizeY>, Int<boxSizeX>>;

            using LayoutA = cutlass::layout::RowMajor;
            using StrideA = cutlass::detail::TagToStrideC_t<LayoutA>;
            using LayoutB = cutlass::layout::RowMajor;
            using StrideB = cutlass::detail::TagToStrideC_t<LayoutB>;
            using SmemLayout = Layout<Shape<Int<boxSizeY>, Int<boxSizeX>>, Stride<Int<boxSizeX>, _1>>;
            
            auto problem_shape = make_shape(gmemSizeY, gmemSizeX, 0, 1);

            using CollectiveEpilogue = cutlass::epilogue::collective::DefaultEpilogue<
                StrideA,
                StrideB,
                SmemLayout,
                SmemLayout,
                TileShape,
#if defined(RELU)
                DMAPostOPReLu<dtype_dst, dtype_src, 32, 4, 16>,
#elif defined(CONVERSION)
                DMAPostOPConvert<dtype_dst, dtype_src, 32, 4, 16>,
#endif
                cutlass::gemm::EpilogueDefault
            >;

            using ElementSrc = typename CollectiveEpilogue::ElementC;
            using SmemLayoutSrc = typename CollectiveEpilogue::SmemLayoutC;
            using GmemTiledCopySrc = typename CollectiveEpilogue::GmemTiledCopyC;
            using AuxParamsSrc = AuxParams<slm_matrix_type::type1, typename CollectiveEpilogue::TensorDescPtr, 0>;
            using SrcLoadPipeline = typename CollectiveEpilogue::EpilogueLoadPipeline;
            using SrcLoadPipelineState = typename CollectiveEpilogue::LoadPipelineState;

            struct SharedStorage
            {
                using EpilogueTensorStorage = typename CollectiveEpilogue::TensorStorage;

                struct TensorStorage
                {
                    EpilogueTensorStorage epilogue;
                } tensors;

                cute::array<ElementSrc, cute::cosize_v<SmemLayoutSrc>> src;
            };

            auto ptr = sycl::ext::oneapi::group_local_memory_for_overwrite<uint8_t[sizeof(SharedStorage)]>(item.get_group());
            SharedStorage* shared_storage = reinterpret_cast<SharedStorage*>(*ptr);

            typename CollectiveEpilogue::Arguments args = {
                B_d, cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(gmemSizeY, gmemSizeX, 1)),
            };

            typename CollectiveEpilogue::Params params = CollectiveEpilogue::to_underlying_arguments(problem_shape, args, nullptr);
            uint32_t local_id = item.get_local_linear_id();
            SrcLoadPipeline src_load_pipeline(local_id);
            SrcLoadPipelineState src_load_state = cutlass::xe4::make_producer_start_state<SrcLoadPipeline>();
            typename CollectiveEpilogue::EpilogueStorePipeline epilogue_store_pipeline(local_id);

            CollectiveEpilogue collective_epilogue(params);

            auto blk_coord_mnl = cute::make_tuple(item.get_group(1), item.get_group(2), 0);

            if (local_id == 0) {
                auto [M, N, K, L] = problem_shape;
                auto src = make_tensor(A_d, make_layout(make_shape(M, N, L), cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(gmemSizeY, gmemSizeX, 1))));
                auto load_src = make_xe4_copy<GmemTiledCopySrc, AuxParamsSrc>(src, SmemLayoutSrc {}, make_shape(shape<0>(TileShape {}), shape<1>(TileShape {})));
                auto sSrc = make_tensor(reinterpret_cast<ElementSrc *>(shared_storage->src.data()), SmemLayoutSrc {});
                auto mSrc_mnl = load_src.get_tma_tensor(make_shape(M, N, L)); // (m,n,l)
                auto gSrc_mnl = flat_divide(mSrc_mnl,make_shape(shape<0>(TileShape {}), shape<1>(TileShape {}))); // (BLK_M,BLK_N,m,n,l)
                auto block_load_src = load_src.get_slice(0);
                auto [m_coord, n_coord, l_coord] = blk_coord_mnl;
                auto gSrc = gSrc_mnl(_, _, m_coord, n_coord, 0); // (BLK_M,BLK_N)
                auto tSrcgSrc = block_load_src.partition_S(gSrc); // (TMA,TMA_M,TMA_N)
                auto tSrcsSrc = block_load_src.partition_D(sSrc); // (TMA,TMA_M,TMA_N)

                src_load_pipeline.producer_try_wait(src_load_state);
                auto abar_prod = src_load_pipeline.producer_get_barrier(src_load_state);
                constexpr uint32_t slm_bytes_load = sizeof(ElementSrc) * size(SmemLayoutSrc {});
                copy(load_src.with(abar_prod), tSrcgSrc, tSrcsSrc);
                src_load_pipeline.producer_commit(src_load_state, sizeof(SharedStorage::src));
                src_load_pipeline.consumer_try_wait(src_load_state.index(), 0);
            }

            item.barrier(access::fence_space::local_space);

            if (local_id >= 128) {
                auto tensor_src = make_tensor(
                        reinterpret_cast<ElementSrc *>(shared_storage->src.data()),
                        SmemLayoutSrc {});
                collective_epilogue(tensor_src, shared_storage->tensors.epilogue, local_id);
            }

            item.barrier(access::fence_space::local_space);

            if (local_id == 0) {
                collective_epilogue.store(epilogue_store_pipeline,
                        typename CollectiveEpilogue::StorePipelineState {}, problem_shape,
                        blk_coord_mnl, shared_storage->tensors.epilogue);
            }

        });
     }).wait();

    q.memcpy(B_h.data(), B_d, gmemSize * sizeof(dtype_dst)).wait();

#if defined(RELU)
    ReLu post_op;
#elif defined(CONVERSION)
    Conversion<dtype_dst> post_op;
#endif

    uint32_t err_cnt = 0;
    for (auto i = 0; i != gmemSizeY; i++)
    {
        for (auto j = 0; j != gmemSizeX; j++)
        {
            uint32_t idx = i * gmemSizeX + j;
            if (B_h[idx] != static_cast<dtype_dst>(post_op(A_h[idx])))
            {
                err_cnt++;
                std::cout << " B: " << B_h[idx]
                          << " mismatch with A: " << static_cast<dtype_dst>(post_op(A_h[idx]))
                          << " at idx = " << idx << std::endl;
            }
        }
    }
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
    free(A_d, q);
    free(B_d, q);

    return rtn;
}
