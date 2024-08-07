#pragma once

#include "cutlass/cutlass.h"

using namespace cute;

/// Converts the result without other operations
template <
  typename TileShape_,
  typename ElementOutput_,
  typename ElementAccumulator_
>
class DummyConverter {
public:
  using TileShape = TileShape_;
  using ElementOutput = ElementOutput_;
  using ElementAccumulator = ElementAccumulator_;
  using SmemLayoutOutput = decltype(make_layout(TileShape{}, GenRowMajor{}));
  using TensorOutput = decltype(make_tensor(static_cast<ElementOutput*>(nullptr), SmemLayoutOutput {}));
  using TensorAccumulator= decltype(make_tensor(static_cast<ElementAccumulator*>(nullptr), SmemLayoutOutput {}));

  struct Params {
    CUTLASS_HOST_DEVICE
    Params() {}
  };

public:

  CUTLASS_HOST_DEVICE
  DummyConverter(Params const &params = Params()) {
  }

  CUTLASS_HOST_DEVICE
  void operator()(
    TensorAccumulator const &accumulator,
    TensorOutput &output) const {
  }
};
