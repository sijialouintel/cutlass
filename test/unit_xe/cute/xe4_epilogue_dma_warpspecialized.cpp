#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cutlass/epilogue/collective/xe4_epilogue_dma_warpspecialized.hpp>
#include <cutlass/epilogue/thread/xe4_conversion_op.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/util/packed_stride.hpp>

#include <CL/sycl.hpp>

using namespace cute;
using namespace cute::xe4;
using namespace sycl;
using namespace cutlass::epilogue::thread;
using namespace cutlass::epilogue::thread::detail;

int main()
{
    using SrcType = uint32_t;
    using DstType = uint16_t;
    constexpr int32_t boxSizeX = 32;
    constexpr int32_t boxSizeY = 32;
    using TileShape = Shape<Int<boxSizeY>, Int<boxSizeX>>;
    using LayoutA = cutlass::layout::RowMajor;
    using StrideA = cutlass::detail::TagToStrideC_t<LayoutA>;
    using LayoutB = cutlass::layout::RowMajor;
    using StrideB = cutlass::detail::TagToStrideC_t<LayoutB>;
    using SmemLayout = Layout<Shape<Int<boxSizeY>, Int<boxSizeX>>, Stride<Int<boxSizeX>, _1>>;
    using CollectiveEpilogue = cutlass::epilogue::collective::DefaultEpilogue<
                StrideA,
                StrideB,
                SmemLayout,
                SmemLayout,
                TileShape,
                DMAPostOPConvert<DstType, SrcType, 32, 4, 16, EpilogueAccessPattern::Pattern2>,
                cutlass::gemm::EpilogueDefault
            >;

    SrcType src[boxSizeY * boxSizeX] {};
    DstType dst[boxSizeY * boxSizeX] {};
    typename CollectiveEpilogue::Arguments args = {
        dst, cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(boxSizeY, boxSizeX, 1)),
    };

    auto problem_shape = make_shape(boxSizeY, boxSizeX, 0, 1);
    typename CollectiveEpilogue::Params params = CollectiveEpilogue::to_underlying_arguments(problem_shape, args, nullptr);
    auto src_tensor = make_tensor((SrcType*)src, SmemLayout {});

    CollectiveEpilogue::TensorStorage dst_tensor;

    CollectiveEpilogue collective_epilogue(params);
    collective_epilogue(src_tensor, dst_tensor, 128);
    return 0;
}
