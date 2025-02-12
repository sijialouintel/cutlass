#pragma once

#include "cutlass/arch/barrier_xe4.h"

namespace cutlass::xe4 {

namespace detail {

// Helper function for DEBUG checks
template<class ThreadCategory>
CUTLASS_DEVICE
bool pipeline_is_producer(ThreadCategory role) {
  return (role == ThreadCategory::Producer || role == ThreadCategory::ProducerConsumer);
}

template<class ThreadCategory>
CUTLASS_DEVICE
void pipeline_check_is_producer(ThreadCategory role) {
}

template<class ThreadCategory>
CUTLASS_DEVICE
bool pipeline_is_consumer(ThreadCategory role) {
  return (role == ThreadCategory::Consumer || role == ThreadCategory::ProducerConsumer);
}

template<class ThreadCategory>
CUTLASS_DEVICE
void pipeline_check_is_consumer(ThreadCategory role) {
}

} // namespace detail

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
  PipelineState operator++(int) {
    PipelineState temp = *this;
    ++(*this);
    return temp;
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

template <int Stages_>
class PipelineTmaAsync {
public:
  using FullBarrier = cutlass::arch::xe4::AddressableTransactionBarrier;
  using EmptyBarrier = cutlass::arch::xe4::AddressableTransactionBarrier;
  using ProducerBarrierType = FullBarrier::ValueType;
  using ConsumerBarrierType = EmptyBarrier::ValueType;
  static constexpr int Stages = Stages_;
  using PipelineState = cutlass::xe4::PipelineState<Stages>;

  struct SharedStorage {
    FullBarrier full_barrier_[Stages];
    EmptyBarrier empty_barrier_[Stages];
  };

  enum class ThreadCategory {
    NonParticipant,
    Producer,
    Consumer,
    ProducerConsumer
  };

  struct Params {
    uint32_t transaction_bytes = 0;
    ThreadCategory role = ThreadCategory::NonParticipant;
    uint32_t is_leader = 0;
    uint32_t num_consumers = 1; // Number of consumer threads
    uint32_t num_producers = 1; // Number of producer threads
    uint32_t local_id = 0;
    uint32_t initializing_id = 0;
  };

  static
  CUTLASS_DEVICE
  void
  init_barriers(SharedStorage& storage, Params params) {
    if (params.local_id == 0) {
      // Barrier FULL and EMPTY init
      uint32_t const producer_arv_cnt = params.num_producers;
      uint32_t const consumer_arv_cnt = params.num_consumers;
      cutlass::arch::xe4::detail::initialize_barrier_array_pair_aligned<decltype(storage.full_barrier_), decltype(storage.empty_barrier_), Stages>(
          storage.full_barrier_, storage.empty_barrier_, producer_arv_cnt, consumer_arv_cnt);
    }
  }

  CUTLASS_DEVICE
  PipelineTmaAsync(SharedStorage& storage, Params params)
    : full_barrier_ptr_(&storage.full_barrier_[0])
    , empty_barrier_ptr_(&storage.empty_barrier_[0]) {
    init_barriers(storage, params);
  }

  ////////////////////
  // Producer APIs
  ////////////////////

  CUTLASS_DEVICE
  void producer_try_wait(PipelineState state, uint32_t skip_wait = false) {
    return producer_try_wait(state.index(), state.phase(), skip_wait);
  }

  CUTLASS_DEVICE
  void producer_commit(PipelineState state, uint32_t bytes) {
    producer_commit(state.index(), bytes);
  }

  CUTLASS_DEVICE
  void producer_arrive(PipelineState state, uint32_t bytes) {
    producer_arrive(state.index(), bytes);
  }

  CUTLASS_DEVICE
  ProducerBarrierType* producer_get_barrier(PipelineState state) {
    return producer_get_barrier(state.index());
  }

  ////////////////////
  // Consumer APIs
  ////////////////////

  CUTLASS_DEVICE
  void consumer_try_wait(PipelineState state, uint32_t skip_wait = false) {
    consumer_try_wait(state.index(), state.phase(), skip_wait);
  }

  CUTLASS_DEVICE
  void consumer_commit(PipelineState state, uint32_t count=1) {
    consumer_commit(state.index(), count);
  }

  CUTLASS_DEVICE
  void consumer_arrive(PipelineState state, uint32_t bytes) {
    consumer_arrive(state.index(), bytes);
  }

  CUTLASS_DEVICE
  ConsumerBarrierType* consumer_get_barrier(PipelineState state) {
    return consumer_get_barrier(state.index());
  }

private:
  FullBarrier *full_barrier_ptr_ = nullptr;
  EmptyBarrier *empty_barrier_ptr_ = nullptr;

  CUTLASS_DEVICE
  void producer_try_wait(uint32_t stage, uint32_t phase, uint32_t skip_wait = false) {
    empty_barrier_ptr_[stage].try_wait(phase);
  }

  CUTLASS_DEVICE
  void producer_commit(uint32_t stage, uint32_t bytes) {
    full_barrier_ptr_[stage].arrive_and_expect_tx(bytes);
  }

  CUTLASS_DEVICE
  void producer_arrive(uint32_t stage, uint32_t bytes) {
    full_barrier_ptr_[stage].arrive(bytes);
  }

  CUTLASS_DEVICE
  ProducerBarrierType* producer_get_barrier(uint32_t stage) {
    return reinterpret_cast<ProducerBarrierType*>(&full_barrier_ptr_[stage]);
  }

  CUTLASS_DEVICE
  void consumer_try_wait(uint32_t stage, uint32_t phase, uint32_t skip_wait = 0) {
    full_barrier_ptr_[stage].try_wait(phase);
  }

  CUTLASS_DEVICE
  void consumer_commit(uint32_t stage, uint32_t count, uint32_t skip = false) {
    empty_barrier_ptr_[stage].arrive_and_expect_tx(count);
  }

  CUTLASS_DEVICE
  void consumer_arrive(uint32_t stage, uint32_t bytes) {
    empty_barrier_ptr_[stage].arrive(bytes);
  }

  CUTLASS_DEVICE
  ConsumerBarrierType* consumer_get_barrier(uint32_t stage) {
    return reinterpret_cast<ProducerBarrierType*>(&empty_barrier_ptr_[stage]);
  }
};

} // namespace cutlass::xe4
