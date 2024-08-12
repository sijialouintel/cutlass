#!/usr/bin/env bash

export ZESIM_ROOT=/root/zesim/debug/zesim

source /opt/intel/oneapi/setvars.sh
export IGC20_ROOT=/root/llc
export IGC_DumpToCustomDir=./igc_dump
export IGC_ShaderDumpEnable=1
export NEO_CACHE_PERSISTENT=0
export IGC_EnableEfficient64b=1
export IGC_VCInternalOptions=-ze-intel-64bit-addressing

rm -rf build; mkdir build; cd build

SCRIPT_PATH=$(dirname "$(realpath "$0")")
XE4_TEST_PATH=$(realpath "$SCRIPT_PATH/../../../3rdparty/xe4_test")
CUTLASS_PISA_PATH=$(realpath "$SCRIPT_PATH/../../../../..")

INCLUDE_PATHS="-I$CUTLASS_PISA_PATH/include \
               -I/usr/local/cuda/include \
               -I$XE4_TEST_PATH/pisa_tests \
               -I$CUTLASS_PISA_PATH/tools/util/include \
               -I$CUTLASS_PISA_PATH/test/unit_xe/functional/bgemm"

icpx -fsycl -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm $INCLUDE_PATHS ../bgemm.cpp -o bgemm

export LD_LIBRARY_PATH=$ZESIM_ROOT:$LD_LIBRARY_PATH
export L0SIM_DEVICE_KIND=Xe4
export L0SIM_GRITS_PATH=/root/FCS

export L0SIM_SELECT_DEVICES=XE4ISAI
# export L0SIM_SELECT_DEVICES=GRITS

# export XE4_KERNEL_BINARY_REPLACE_PATH=./
export XE4_LOG_ON="1"
#export XE4_LOG_ON="1" # set to 0 to turn log off, default off
# default log folder "dump"
export XE4_LOG_FOLDER_PATH="./logdump"

./bgemm

find build -type f -name "*.pisa" | while read -r file; do
    sed -i '/Inline assembly/d' "$file"
done

