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
  --shape 64x256x128 \
  --dtype bf16_bf16_bf16 f32_f32_bf16_bf16 f32_bf16_bf16 bf16_f32_bf16_bf16

test_cases=("SMALL")


if [ "$#" -gt 0 ] && [[ "$1" =~ ^[0-9]+$ ]]; then
    DEBUG_TID="$1"
else
    DEBUG_TID="${DEBUG_TID:-0}"
fi


build_test_case() {
  local test_case=$1
  echo "Building $test_case"
  local work_dir=$BUILD_PATH/$test_case
  mkdir -p $work_dir
  cd $work_dir
  icpx -fsycl -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldnnl \
    -DTEST_$test_case -DAMMA_HEADER_PATH=$GEN_HEADER_PATH \
    -DDRY_RUN -DDEBUG_TID=$DEBUG_TID \
    $INCLUDE_PATHS $ORIGIN_PATH/conv2d.cpp -o $work_dir/conv2d
  cd $ORIGIN_PATH
}
# -DTRACE_PISA

run_test_case() {
  local test_case=$1
  local work_dir=$BUILD_PATH/$test_case
  cd $work_dir
  ./conv2d
  cd $ORIGIN_PATH
}

for ((i=0; i<1; i++)); do
  build_test_case ${test_cases[i]}
  run_test_case ${test_cases[i]}
done

