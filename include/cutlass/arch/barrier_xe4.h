#pragma once

#include "inline_pisa.hpp"

#ifndef CUTLASS_HOST_DEVICE
#define CUTLASS_HOST_DEVICE inline
#endif

#ifndef CUTLASS_DEVICE
#define CUTLASS_DEVICE inline
#endif

namespace cutlass {
/// @brief
namespace arch {
namespace xe4 {


namespace detail { // namespace detail begin
template<typename FullBarrier, typename EmptyBarrier, uint32_t Stages>
CUTLASS_DEVICE
void initialize_barrier_array_pair_aligned(FullBarrier full_barriers, EmptyBarrier empty_barriers, int full_barrier_arv_cnt, int empty_barrier_arv_cnt) {
  if(cute::elect_one_sync()) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Stages; i++) {
      full_barriers[i].init(full_barrier_arv_cnt);
      empty_barriers[i].init(empty_barrier_arv_cnt);
    }
  }
}

} // namespace detail end

// xe4 introduces addressable barrier, which is a HW supported synchronization object
// to coordinate execution and data sharing across EU threads and async operations.
struct AddressableBarrier {
  using ValueType = uint64_t;

protected:
  // Can never be initialized - can only be aliased to smem
  ValueType barrier_;

public:
  CUTLASS_DEVICE
  AddressableBarrier() = delete;

  CUTLASS_DEVICE
  void init(uint32_t arrive_count) const {
    AddressableBarrier::init(&this->barrier_, arrive_count);
  }

  CUTLASS_DEVICE
  void try_wait(uint32_t phase) const {
    AddressableBarrier::try_wait(&this->barrier_, phase);
  }

  // Barrier arrive on local smem
  CUTLASS_DEVICE
  void arrive(uint32_t arrive_cnt) const {
    AddressableBarrier::arrive(&this->barrier_, arrive_cnt);
  }

  //
  //  Static Versions
  //
  CUTLASS_DEVICE
  static void init(ValueType const* abar_ptr, uint32_t arrive_count) {
    abarrier_init(abar_ptr, arrive_count);
  }

  CUTLASS_DEVICE
  static void try_wait(ValueType const* abar_ptr, uint32_t phase_bit) {
    abarrier_try_wait(abar_ptr, phase_bit);
  }

  CUTLASS_DEVICE
  static void arrive(ValueType const* abar_ptr, uint32_t arrive_cnt) {
    abarrier_workgroup_arrive(abar_ptr, arrive_cnt);
  }
};


////////////////////////////////////////////////////////////////////////////////////////////////////

// xe4 also introduces a new type of addressable-barrier which supports sync.
// not just based on Arrive Count, but also transaction count (in bytes)
struct AddressableTransactionBarrier : public AddressableBarrier {
  CUTLASS_DEVICE
  AddressableTransactionBarrier() = delete;

  // Performs an arrive operation + expected transaction bytes increment
  CUTLASS_DEVICE
  void arrive_and_expect_tx(uint32_t transaction_bytes) const {
    AddressableTransactionBarrier::arrive_and_expect_tx(&this->barrier_, transaction_bytes);
  }

  //
  //  Static Versions
  //

  // Performs an arrive operation + expected transaction bytes increment
  CUTLASS_DEVICE
  static void arrive_and_expect_tx(ValueType const* abar_ptr, uint32_t transaction_bytes) {
    abarrier_workgroup_arrive_expect_tx(abar_ptr, transaction_bytes);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
}  // end namespace xe4
}  // end namespace arch
}  // end namespace cutlass
