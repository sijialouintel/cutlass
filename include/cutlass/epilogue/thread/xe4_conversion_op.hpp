#pragma once

#include "cute/tensor.hpp"
#include "cute/util/print.hpp"
#include "cutlass/epilogue/thread/xe4_detail.hpp"

template <typename DstType>
struct Conversion {
  template <class SrcType>
  CUTLASS_HOST_DEVICE DstType operator()(SrcType value) const {
    return static_cast<DstType>(value);
  }
};

namespace cutlass {
namespace epilogue {
namespace thread {

using namespace cute;
using namespace cutlass::epilogue::thread::detail;

template <
  typename ElementOutput_,
  typename ElementAccumulator_,
  uint32_t SubGroupSize_,
  uint32_t NumControlSubGroup_,
  uint32_t NumPostOpSubGroup_,
  EpilogueAccessPattern AccessPattern_
>
class DMAPostOPConvert {
public:
  using ElementOutput = ElementOutput_;
  using ElementAccumulator = ElementAccumulator_;

  static constexpr uint32_t SubGroupSize = SubGroupSize_;
  static constexpr uint32_t NumControlSubGroup = NumControlSubGroup_;
  static constexpr uint32_t NumPostOpSubGroup = NumPostOpSubGroup_;
  static constexpr EpilogueAccessPattern AccessPattern = AccessPattern_;

  static_assert(AccessPattern == EpilogueAccessPattern::Pattern2, "Unsupportted access pattern!");

  /**
   * @brief Perform data conversion.
   *
   * @tparam SrcTensor Src tensor type, composition of swizzle and core matrix grid layout.
   * @tparam DstTensor Same as above but for dst tensor type.
   * @param src_tensor
   * @param dst_tensor
   * @param local_id Work item local linear id.
   */
  template <class SrcTensor, class DstTensor>
  void operator()(SrcTensor const &src_tensor, DstTensor &dst_tensor, uint32_t local_id) const {
    HOST_PRINT(src_tensor);
    HOST_PRINT(dst_tensor);
    uint32_t worker_id = local_id - NumControlSubGroup * SubGroupSize;
    if constexpr (AccessPattern == EpilogueAccessPattern::Pattern2) {
      pattern2<Conversion<ElementOutput>, NumPostOpSubGroup, SubGroupSize>(src_tensor, dst_tensor, worker_id);
    }
  }
};

} // namespace thread
} // namespace epilogue
} // namespace cutlass
