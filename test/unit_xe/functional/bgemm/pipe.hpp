#pragma once

#include "inline_pisa.hpp"

#ifndef CUTLASS_HOST_DEVICE
#define CUTLASS_HOST_DEVICE inline
#endif

#ifndef CUTLASS_DEVICE
#define CUTLASS_DEVICE inline
#endif

namespace cutlass::xe4 {

// Circular Buffer Index + Associated Phase
// Assumes only one operation possible - i.e., ++
template<uint32_t Stages_>
struct PipelineState {

  static constexpr uint32_t Stages = Stages_;

  int index_ = 0;
  uint32_t phase_ = 0;
  uint32_t count_ = 0;

  CUTLASS_DEVICE
  PipelineState(): index_{}, phase_{}, count_{} {}

  CUTLASS_DEVICE
  PipelineState(int index, uint32_t phase, uint32_t count)
    : index_(index)
    , phase_(phase)
    , count_(count) {}

  CUTLASS_DEVICE
  int index() const {
    return index_;
  }

  CUTLASS_DEVICE
  uint32_t phase() const {
    return phase_;
  }

  CUTLASS_DEVICE
  uint32_t count() const {
    return count_;
  }

  CUTLASS_DEVICE
  void operator++() {
    if constexpr (Stages > 0) {
      ++index_;
      ++count_;
      if (index_ == Stages) {
        index_ = 0;
        phase_ ^= 1;
      }
    }
  }

  CUTLASS_DEVICE
  PipelineState& operator+=(uint32_t num_iterations) {
    return advance(num_iterations);
  }

  CUTLASS_DEVICE
  PipelineState& operator=(PipelineState const& other) {
    index_ = other.index();
    phase_ = other.phase();
    count_ = other.count();
    return *this;
  }

  CUTLASS_DEVICE
  PipelineState& advance(uint32_t num_iterations) {
    if constexpr (Stages > 0) {
      // Number of iterations cross over the stage boundary => flipped phase
      if ((num_iterations < Stages) && (index_ + num_iterations) >= Stages ) {
        phase_ ^= 1;
      }
      // How many times number of iterations cross over the stage boundary and
      // end up on a odd number => flipped phase
      if ((num_iterations >= Stages) && (((index_ + num_iterations) / Stages) % 2) == 1) {
        phase_ ^= 1;
      }
      index_ = (index_ + num_iterations) % Stages;
      count_ += num_iterations;
    }
    return *this;
  }

  CUTLASS_DEVICE
  static PipelineState make_pipeline_state(PipelineState start_state, uint32_t num_iterations) {
    return start_state.advance(num_iterations);
  }
};

template<class Pipeline>
CUTLASS_DEVICE
PipelineState<Pipeline::Stages> make_producer_start_state() {
  // Producer starts with an opposite phase as the buffers are initially empty
  constexpr int InitialProducerStage = 0;
  constexpr uint32_t InitialProducerPhase = 1;
  constexpr uint32_t InitialProducerCount = 0;
  return {InitialProducerStage, InitialProducerPhase, InitialProducerCount};
}

template <int Stages_, typename BarrierPtr = uint64_t*>
class PipelineTmaAsync {
public:
  using ProducerBarrier = BarrierPtr;
  using ConsumerBarrier = BarrierPtr;
  static constexpr int Stages = Stages_;
  using PipelineState = cutlass::xe4::PipelineState<Stages>;

  BarrierPtr abar_prod_base = nullptr;
  BarrierPtr abar_cons_base = nullptr;

  PipelineTmaAsync(sycl::nd_item<3> item) {
    uint32_t local_id = item.get_local_linear_id();
    abar_prod_base = allocate_abar<0,Stages>();
    abar_cons_base = allocate_abar<1,Stages>();
    if (local_id == 0) {
      #pragma unroll
      for (int i = 0; i < Stages; i++) {
        abarrier_init(abar_prod_base + i, 1);
      }
    } else if (local_id == 32) {
      #pragma unroll
      for (int i = 0; i < Stages; i++) {
        abarrier_init(abar_cons_base + i, 1);
      }
    }
    item.barrier(access::fence_space::local_space);
  }

  CUTLASS_DEVICE
  void producer_try_wait(PipelineState state, uint32_t skip_wait = false) {
    return producer_try_wait(state.index(), state.phase(), skip_wait);
  }

  CUTLASS_DEVICE
  void producer_try_wait(uint32_t stage, uint32_t phase, uint32_t skip_wait = false) {
    abarrier_try_wait(abar_cons_base + stage, phase);
  }

  CUTLASS_DEVICE
  void producer_commit(PipelineState state, uint32_t bytes) {
    producer_commit(state.index(), bytes);
  }

  CUTLASS_DEVICE
  void producer_commit(uint32_t stage, uint32_t bytes) {
    abarrier_workgroup_arrive_expect_tx(abar_prod_base + stage, bytes);
  }

  CUTLASS_DEVICE
  void consumer_try_wait(PipelineState state, uint32_t skip_wait = false) {
    consumer_try_wait(state.index(), state.phase(), skip_wait);
  }

  CUTLASS_DEVICE
  void consumer_try_wait(uint32_t stage, uint32_t phase, uint32_t skip_wait = 0) {
    abarrier_try_wait(abar_prod_base + stage, phase);
  }

  CUTLASS_DEVICE
  void consumer_commit(PipelineState state) {
    consumer_commit(state.index());
  }

  CUTLASS_DEVICE
  void consumer_commit(uint32_t stage, uint32_t skip = false) {
    abarrier_workgroup_arrive_expect_tx(abar_cons_base + stage, 1);
  }

  CUTLASS_DEVICE
  ProducerBarrier producer_get_barrier(PipelineState state) {
    return producer_get_barrier(state.index());
  }

  CUTLASS_DEVICE
  ProducerBarrier producer_get_barrier(uint32_t stage) {
    return abar_prod_base + stage;
  }

  CUTLASS_DEVICE
  ConsumerBarrier consumer_get_barrier(PipelineState state) {
    return consumer_get_barrier(state.index());
  }

  CUTLASS_DEVICE
  ConsumerBarrier consumer_get_barrier(uint32_t stage) {
    return abar_cons_base + stage;
  }
};

} // namespace cutlass::xe4
