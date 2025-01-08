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
CUTLASS_PISA_PATH=$(realpath "$SCRIPT_PATH/../../../..")
XE4_ROOT="$CUTLASS_PISA_PATH/test/unit_xe/3rdparty/drivers.gpu.compute.workloads/simt_workloads"

INCLUDE_PATHS="-I$CUTLASS_PISA_PATH/include \
  -I/usr/local/cuda/include \
  -I$XE4_ROOT/common_headers \
  -I$CUTLASS_PISA_PATH/tools/util/include"

ORIGIN_PATH=$(pwd)
BUILD_PATH=$ORIGIN_PATH/build

rm -rf $BUILD_PATH; mkdir -p $BUILD_PATH; cd $BUILD_PATH

GEN_HEADER_PATH=$BUILD_PATH/generated_headers/async_gmma.hpp

python3 $XE4_ROOT/scripts/generator/gen_mma.py \
  --mma_type mma \
  --output $GEN_HEADER_PATH \
  --shape 128x128x128 256x512x128 --dtype f32_f32_bf16_bf16 bf16_f32_bf16_bf16

icpx -fsycl -std=c++20 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -Xs " -xe-set-abarrier-arrive-lmc " \
  -DTEST_$test_case -DAMMA_HEADER_PATH=$GEN_HEADER_PATH $INCLUDE_PATHS $ORIGIN_PATH/bgemm.cpp -o bgemm

export L0SIM_DEVICE_KIND=Xe4
export L0SIM_GRITS_PATH=/root/XE3P_V2
# export L0SIM_SELECT_DEVICES=XE4ISAI
export L0SIM_SELECT_DEVICES=GRITS

export XE4_LOG_ON="1"
export XE4_LOG_FOLDER_PATH="./logdump"
export ZESIM_ROOT=/root/zesim/debug/zesim
export LD_LIBRARY_PATH=$ZESIM_ROOT:$LD_LIBRARY_PATH

./bgemm

find . -type f -name "*.pisa" | while read -r file; do
  sed -i '/Inline assembly/d' "$file"
done
