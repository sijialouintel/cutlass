#pragma once

#include "cute/tensor.hpp"
#include "cute/util/print.hpp"
#include "cute/numeric/numeric_types.hpp"
#include "cutlass/epilogue/thread/xe4_detail.hpp"

template <typename DstType>
struct Conversion {
  template <class SrcType>
  CUTLASS_HOST_DEVICE DstType operator()(SrcType value) const {
    return static_cast<DstType>(value);
  }

  template <class EngineIn, class LayoutIn,
            class EngineOut, class LayoutOut>
  CUTE_HOST_DEVICE constexpr
  void
  transform(cute::Tensor<EngineIn, LayoutIn > const& tensor_in,
            cute::Tensor<EngineOut,LayoutOut>      & tensor_out)
  {
    using OutType = typename EngineOut::value_type;
    using PackType = cute::uint_bit_t<2 * cute::sizeof_bits_v<OutType>>;
    auto packed_out = recast<PackType>(tensor_out);

    CUTE_UNROLL
    for (int i = 0; i < size(packed_out); ++i) {
      cvt_pack<OutType>(packed_out(i), tensor_in(2*i), tensor_in(2*i+1));
    }
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
  uint32_t SubgroupSize_,
  uint32_t NumControlSubgroup_,
  uint32_t NumPostOpSubgroup_,
  EpilogueAccessPattern AccessPattern_
>
class DMAPostOPConvert {
public:
  using ElementOutput = ElementOutput_;
  using ElementAccumulator = ElementAccumulator_;

  static constexpr uint32_t SubgroupSize = SubgroupSize_;
  static constexpr uint32_t NumControlSubgroup = NumControlSubgroup_;
  static constexpr uint32_t NumPostOpSubgroup = NumPostOpSubgroup_;
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
    uint32_t worker_id = local_id - NumControlSubgroup * SubgroupSize;
    if constexpr (AccessPattern == EpilogueAccessPattern::Pattern2) {
      pattern2<Conversion<ElementOutput>, NumPostOpSubgroup, SubgroupSize>(src_tensor, dst_tensor, worker_id);
    }
  }
};

} // namespace thread
} // namespace epilogue
} // namespace cutlass
