#pragma once

#include <algorithm>
#include <vector>

#include "cute/tensor.hpp"
#include "cute/util/print.hpp"
#include "cutlass/epilogue/thread/xe4_detail.hpp"

struct ReLu {
  template <typename T>
  CUTLASS_HOST_DEVICE T operator()(T value) const {
    return sycl::fmax(value, T(0));
  }
  /* Use for calculating golden value. */
  template <class T>
  void run(std::vector<T> &vec) {
    for (auto &val : vec) {
      val = operator()(val);
    }
  }
};

namespace cutlass {
namespace epilogue {
namespace thread {

using namespace cutlass::epilogue::thread::detail;

template <
  typename ElementOutput_,
  typename ElementAccumulator_,
  uint32_t SubgroupSize_,
  uint32_t NumControlSubgroup_,
  uint32_t NumPostOpSubgroup_,
  EpilogueAccessPattern AccessPattern_
>
class DMAPostOPReLu {
public:
  using ElementOutput = ElementOutput_;
  using ElementAccumulator = ElementAccumulator_;

  static constexpr uint32_t SubgroupSize = SubgroupSize_;
  static constexpr uint32_t NumControlSubgroup = NumControlSubgroup_;
  static constexpr uint32_t NumPostOpSubgroup = NumPostOpSubgroup_;
  static constexpr EpilogueAccessPattern AccessPattern = AccessPattern_;

  static_assert(AccessPattern == EpilogueAccessPattern::Pattern2, "Unsupportted access pattern!");

  /**
   * @brief Perform ReLu.
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
      pattern2<ReLu, NumPostOpSubgroup, SubgroupSize>(src_tensor, dst_tensor, worker_id);
    }
  }
};

} // namespace thread
} // namespace epilogue
} // namespace cutlass
