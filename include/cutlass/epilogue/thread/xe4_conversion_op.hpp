#include <cute/tensor.hpp>

#include <CL/sycl.hpp>

using namespace sycl;
using namespace cute;

constexpr uint32_t const_log2(uint32_t n) {
    uint32_t log = 0;
    uint32_t value = n;
    while (value >>= 1) {
        ++log;
    }
    return log;
}

template<
    class TileShape_,
    class ElementOutput_,
    class ElementAccumulator_,
    uint32_t SubGroupNum
>
class DMAPostOPConvert {
public:
    using TileShape = TileShape_;
    using ElementOutput = ElementOutput_;
    using ElementAccumulator = ElementAccumulator_;

    using SmemLayoutOutput = decltype(make_layout(TileShape{}, GenRowMajor{}));
    using DstTensor = decltype(make_tensor(static_cast<ElementOutput*>(nullptr), SmemLayoutOutput {}));
    using SrcTensor= decltype(make_tensor(static_cast<ElementAccumulator*>(nullptr), SmemLayoutOutput {}));

    template <class SrcTensor, class DstTensor>
    void operator()(SrcTensor const &tensor_src, DstTensor &tensor_dst, uint32_t num_control_sg, uint32_t local_id) const {

        using dtype_src = typename SrcTensor::reference;
        using dtype_dst = typename DstTensor::reference;
        
        constexpr uint32_t boxSizeY = size<0>(SrcTensor{});
        constexpr uint32_t boxSizeX = size<1>(SrcTensor{});
        constexpr uint32_t sg_size = 32;
        uint32_t worker_id = local_id - num_control_sg * sg_size;
        uint32_t sg_id = worker_id / sg_size;
        uint32_t lane_id = worker_id % sg_size;
        constexpr uint32_t src_cm_size_x = 32 / sizeof(dtype_src);
        constexpr uint32_t dst_cm_size_x = 32 / sizeof(dtype_dst);
        // for current design, each item works for 32B on one row of dst core matrix iteratively
        static_assert(boxSizeY * boxSizeX / (SubGroupNum * sg_size) >= dst_cm_size_x);
        constexpr uint32_t core_tile_size = 64;
        constexpr uint32_t core_tile_offset = 256;
        constexpr uint32_t cm_size_y = 32;
        constexpr uint32_t cm_num_src_x = boxSizeX / src_cm_size_x;
        constexpr uint32_t cm_num_dst_x = boxSizeX / dst_cm_size_x;
        constexpr uint32_t cm_num_y = boxSizeY / cm_size_y;
        constexpr uint32_t cm_size_src = cm_size_y * src_cm_size_x;
        constexpr uint32_t cm_size_dst = cm_size_y * dst_cm_size_x;
        constexpr uint32_t cm_rows_per_bank = 2;
        constexpr uint32_t slm_rows_per_cm = 4;
        constexpr uint32_t slm_bank = 4;
        constexpr auto cm_row_shape = make_shape(Int<cm_rows_per_bank>{}, Int<slm_rows_per_cm>{}, Int<slm_bank>{});
        constexpr auto cm_row_stride_src = make_stride(Int<src_cm_size_x>{}, Int<core_tile_offset / sizeof(dtype_src)>{}, Int<core_tile_size / sizeof(dtype_src)>{});
        constexpr auto cm_row_stride_dst = make_stride(Int<dst_cm_size_x>{}, Int<core_tile_offset / sizeof(dtype_dst)>{}, Int<core_tile_size / sizeof(dtype_dst)>{});

        // make tensors for element-wise
        auto sSrc_shape = make_shape(Int<src_cm_size_x>{}, cm_row_shape, make_shape(Int<cm_num_src_x>{}, Int<cm_num_y>{}));
        auto sDst_shape = make_shape(Int<dst_cm_size_x>{}, cm_row_shape, make_shape(Int<cm_num_dst_x>{}, Int<cm_num_y>{}));
        auto sSrc_stride = make_stride(Int<1>{}, cm_row_stride_src, make_stride(Int<cm_size_src>{}, Int<cm_num_src_x * cm_size_src>{}));
        auto sDst_stride = make_stride(Int<1>{}, cm_row_stride_dst, make_stride(Int<cm_size_dst>{}, Int<cm_num_dst_x * cm_size_dst>{}));
        auto sSrc_layout = make_layout(sSrc_shape, sSrc_stride);
        auto sDst_layout = make_layout(sDst_shape, sDst_stride);
        constexpr uint32_t log2_elem_per_cm_x_src = const_log2(src_cm_size_x);
        constexpr uint32_t log2_elem_per_cm_x_dst = const_log2(dst_cm_size_x);
        constexpr uint32_t log2_elem_per_cm_y = const_log2(cm_size_y);
        auto sSrc_layout_s = composition(sSrc_layout, Swizzle<1, log2_elem_per_cm_x_src, log2_elem_per_cm_y>{});
        auto sDst_layout_s = composition(sDst_layout, Swizzle<1, log2_elem_per_cm_x_dst, log2_elem_per_cm_y>{});

        auto sSrc = make_tensor(tensor_src.data(), sSrc_layout_s);
        auto sDst = make_tensor(tensor_dst.data(), sDst_layout_s);

        // load tile
        constexpr uint32_t combined_cm_num_x = sizeof(dtype_src) / sizeof(dtype_dst);
        auto tile_shape_src = make_tile(Int<src_cm_size_x>{}, cm_row_shape, make_shape(Int<combined_cm_num_x>{}, Int<1>{}));
        auto tile_shape_dst = make_tile(Int<dst_cm_size_x>{}, cm_row_shape, make_shape(Int<1>{}, Int<1>{}));

        // copy slm_tiled to regs_tiled
        for(int i = 0; i < cm_num_y * cm_num_dst_x; i += SubGroupNum){
            auto tile_coord = make_coord(_, make_coord(_, _, _), sg_id + i);
            auto tile_sSrc = local_tile(sSrc, tile_shape_src, tile_coord);
            auto tile_sSrc_v = group_modes<3,rank(tile_sSrc)>(tile_sSrc);
            auto tile_sDst = local_tile(sDst, tile_shape_dst, tile_coord);
            auto tile_sDst_v = group_modes<3,rank(tile_sDst)>(tile_sDst);

            // copy src_slm to src_regs
            auto lane_sSrc = tile_sSrc_v(_, lane_id, _, _);
            auto lane_rSrc = make_tensor_like(lane_sSrc);
            auto lane_sDst = tile_sDst_v(_, lane_id, _, _);
            auto lane_rDst = make_tensor_like(lane_sDst);
            copy(lane_sSrc, lane_rSrc);

            // each lane converts src_regs data to dst_regs
            static_assert(lane_rSrc.size() == lane_rDst.size());
            copy(lane_rSrc, lane_rDst);

            // copy dst_regs to dst_slm
            copy(lane_rDst, lane_sDst);
        }
    }
};