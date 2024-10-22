#include "cutlass/epilogue/thread/xe4_conversion_op.hpp"

using namespace cute;
using namespace cutlass::epilogue::thread::detail;

int main() {
  using SrcType = uint32_t;
  using DstType = uint16_t;

  constexpr int subgroup = 16;
  constexpr int subgroup_size = 32;
  constexpr int tile_m = 256;
  constexpr int tile_n = 512;

  SrcType slm_ptr_src[tile_m * tile_n];
  DstType slm_ptr_dst[tile_m * tile_n];

  constexpr auto tile_mn = make_shape(Int<tile_m>{}, Int<tile_n>{});
  auto src_tensor = make_tensor((SrcType*)slm_ptr_src, CoreMatrix::retile<SrcType>(tile_mn));
  auto dst_tensor = make_tensor((DstType*)slm_ptr_dst, CoreMatrix::retile<DstType>(tile_mn));

  using EpilogueOp = cutlass::epilogue::thread::DMAPostOPConvert<DstType, SrcType, subgroup_size, 4, subgroup, EpilogueAccessPattern::Pattern2>;

  EpilogueOp conversion;
  conversion(src_tensor, dst_tensor, 128);

  return 0;
}
