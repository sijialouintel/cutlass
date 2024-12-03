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

rm -rf $BUILD_PATH; mkdir -p $BUILD_PATH; cd $BUILD_PATH

GEN_HEADER_PATH=$BUILD_PATH/generated_headers/async_gmma.hpp

python3 $XE4_TEST_PATH/generator/gen_mma.py \
  --output $GEN_HEADER_PATH \
  --shape 256x512x256 256x512x512 256x128x256 256x128x512 256x256x256 256x256x512 \
  --dtype f32_f32_bf8_e3m0 f32_bf8_e3m0 bf16_f32_bf8_e3m0 f32_f32_e3m0_e3m0 f32_e3m0_e3m0 f16_f32_e3m0_e3m0 bf16_f32_e3m0_e3m0 bf16_f32_bf8_bf8 f32_bf8_bf8 f32_f32_bf8_bf8

icpx -fsycl -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm \
  -DAMMA_HEADER_PATH=$GEN_HEADER_PATH $INCLUDE_PATHS $ORIGIN_PATH/bgemm_mxfp.cpp -o bgemm_mxfp

export L0SIM_DEVICE_KIND=Xe4
export L0SIM_GRITS_PATH=/root/XE3P_V2
# export L0SIM_SELECT_DEVICES=XE4ISAI
export L0SIM_SELECT_DEVICES=GRITS

export XE4_LOG_ON="1"
export XE4_LOG_FOLDER_PATH="./logdump"
export ZESIM_ROOT=/root/zesim/debug-sys20/zesim
export LD_LIBRARY_PATH=$ZESIM_ROOT:$LD_LIBRARY_PATH

./bgemm_mxfp

find . -type f -name "*.pisa" | while read -r file; do
  sed -i '/Inline assembly/d' "$file"
done
