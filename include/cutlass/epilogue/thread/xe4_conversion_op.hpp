#include <cute/tensor.hpp>
#include <cute/util/print.hpp>

using namespace sycl;
using namespace cute;

template<
    class ElementOutput_,
    class ElementAccumulator_,
    uint32_t SubGroupNum_,
    uint32_t SubGroupSize_
>
class DMAPostOPConvert {
public:
    using ElementOutput = ElementOutput_;
    using ElementAccumulator = ElementAccumulator_;
    static constexpr uint32_t SubGroupNum = SubGroupNum_;
    static constexpr uint32_t SubGroupSize = SubGroupSize_;

    template <class SrcTensor, class DstTensor>
    void operator()(SrcTensor const &tensor_src, DstTensor &tensor_dst, uint32_t num_control_sg, uint32_t local_id) const {
        uint32_t worker_id = local_id - num_control_sg * SubGroupSize;
        uint32_t sg_id = worker_id / SubGroupSize;
        uint32_t lane_id = worker_id % SubGroupSize;

        HOST_PRINT(tensor_src);
        HOST_PRINT(tensor_dst);

        // load tile
        constexpr int combined_src_cm_num_x = sizeof(ElementAccumulator) / sizeof(ElementOutput);
        auto tile_shape_src = make_tile(get<0>(tensor_src.shape()), get<1>(tensor_src.shape()), make_shape(Int<combined_src_cm_num_x>{}, Int<1>{}));
        auto tile_shape_dst = make_tile(get<0>(tensor_dst.shape()), get<1>(tensor_dst.shape()), make_shape(Int<1>{}, Int<1>{}));
        HOST_PRINT(tile_shape_src);
        HOST_PRINT(tile_shape_dst);

        // copy slm_tiled to regs_tiled
        for (int i = 0; i < size<2>(tensor_dst.shape()); i += SubGroupNum) {
            if (sg_id + i >= size<2>(tensor_dst.shape())) {
                continue;
            }

            auto tile_coord = make_coord(_, make_coord(_, _, _), sg_id + i);
            auto tile_sSrc = local_tile(tensor_src, tile_shape_src, tile_coord);
            auto tile_sSrc_v = group_modes<3,rank(tile_sSrc)>(tile_sSrc);
            auto tile_sDst = local_tile(tensor_dst, tile_shape_dst, tile_coord);
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