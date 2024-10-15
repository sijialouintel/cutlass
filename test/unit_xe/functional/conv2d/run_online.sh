#!/usr/bin/env bash

set -e

source /opt/intel/oneapi/setvars.sh

export IGC20_ROOT=/root/llc
export IGC_DumpToCustomDir=./igc_dump
export IGC_ShaderDumpEnable=1
export NEO_CACHE_PERSISTENT=0
export IGC_EnableEfficient64b=1
export IGC_VCInternalOptions=-ze-intel-64bit-addressing

SCRIPT_PATH=$(dirname "$(realpath "$0")")
XE4_TEST_PATH=$(realpath "$SCRIPT_PATH/../../3rdparty/xe4_test")
CUTLASS_PISA_PATH=$(realpath "$SCRIPT_PATH/../../../..")

INCLUDE_PATHS="-I$CUTLASS_PISA_PATH/include \
  -I/usr/local/cuda/include \
  -I$XE4_TEST_PATH/pisa_tests \
  -I$CUTLASS_PISA_PATH/tools/util/include"

ORIGIN_PATH=$(pwd)
BUILD_PATH=$ORIGIN_PATH/build

rm -rf $BUILD_PATH; mkdir -p $BUILD_PATH

GEN_HEADER_PATH=$BUILD_PATH/generated_headers/async_gmma.hpp

python3 $XE4_TEST_PATH/generator/gen_mma.py \
  --output $GEN_HEADER_PATH \
  --shape 128x128x128 \
  --dtype bf16_bf16_bf16 f32_f32_bf16_bf16 f32_bf16_bf16 bf16_f32_bf16_bf16

test_cases=("SMALL" "LARGE" "SMALL_WITH_PAD_WITH_STRIDE" "LARGE_WITH_PAD_WITH_STRIDE" "OTHER_WITH_PAD_WITH_STRIDE" "ASYNMMETRIC_PAD_ASYNMMETRIC_STRIDE" "LARGE_WITH_PAD_WITH_STRIDE_WITH_DILATION" "OTHER_WITH_PAD_WITH_STRIDE_WITH_DILATION" "ASYNMMETRIC_PAD_ASYNMMETRIC_STRIDE_WITH_DILATION")

limit=${1:-${#test_cases[@]}}
limit=$((limit > 0 && limit <= ${#test_cases[@]} ? limit : ${#test_cases[@]}))

build_test_case() {
  local test_case=$1
  echo "Building $test_case"
  local work_dir=$BUILD_PATH/$test_case
  mkdir -p $work_dir
  cd $work_dir
  icpx -fsycl -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldnnl \
    -DTEST_$test_case -DAMMA_HEADER_PATH=$GEN_HEADER_PATH \
    $INCLUDE_PATHS $ORIGIN_PATH/conv2d.cpp -o $work_dir/conv2d
  cd $ORIGIN_PATH
}

run_test_case() {
  local test_case=$1
  local work_dir=$BUILD_PATH/$test_case
  cd $work_dir
  ./conv2d
  cd $ORIGIN_PATH
}

for ((i=0; i<limit; i++)); do
  build_test_case ${test_cases[i]}
done

export L0SIM_DEVICE_KIND=Xe4
export L0SIM_GRITS_PATH=/root/FCS
export L0SIM_SELECT_DEVICES=XE4ISAI
# export L0SIM_SELECT_DEVICES=GRITS

export XE4_LOG_ON="1"
export XE4_LOG_FOLDER_PATH="./logdump"
export ZESIM_ROOT=/root/working_dir/drivers.gpu.simulation.gen-isa-interpreter/build/debug-sys20/runtime/src
export LD_LIBRARY_PATH=$ZESIM_ROOT:$LD_LIBRARY_PATH

for ((i=0; i<limit; i++)); do
  run_test_case ${test_cases[i]}
done

find . -type f -name "*.pisa" | while read -r file; do
  sed -i '/Inline assembly/d' "$file"
done
