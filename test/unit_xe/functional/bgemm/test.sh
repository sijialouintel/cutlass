#!/usr/bin/env bash

###################### [1] Activate oneAPI Environment ######################

source /opt/intel/oneapi/setvars.sh

###################### [2] Generate Xe4 Kernel Binary ######################

export IGC_DumpToCustomDir=./igc_dump
export IGC_ShaderDumpEnable=1
export NEO_CACHE_PERSISTENT=0
export IGC_EnableEfficient64b=1
export IGC_VCInternalOptions=-ze-intel-64bit-addressing
export IGC_20_PATH=/root/llc
export PATH="$IGC_20_PATH/bin:$PATH"

rm -rf build; mkdir build; cd build

EXTRA_DEFS="-DCUTLASS_ENABLE_SYCL"

DPCPP_COMPILER=icpx
SCRIPT_PATH=$(dirname "$(realpath "$0")")
XE4_TEST_PATH=$(realpath "$SCRIPT_PATH/../../../3rdparty/xe4_test")
CUTLASS_PISA_PATH=$(realpath "$SCRIPT_PATH/../../../../..")

INCLUDE_PATHS="-I$CUTLASS_PISA_PATH/include \
               -I/usr/local/cuda/include \
               -I$XE4_TEST_PATH/pisa_tests \
               -I$CUTLASS_PISA_PATH/tools/util/include \
               -I$CUTLASS_PISA_PATH/test/unit_xe/functional"

$DPCPP_COMPILER -fsycl -fsycl-targets=spir64_gen -Xs "-device fcs" -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -std=c++17 -fPIE $INCLUDE_PATHS $EXTRA_DEFS ../bgemm.cpp -o bgemm

for file in igc_dump/*.spv; do
  echo "Processing $file"
  LD_LIBRARY_PATH=$IGC_20_PATH/lib:$LD_LIBRARY_PATH \
  ocloc compile -device fcs -spirv_input -gen_file -output_no_suffix -file $file 2>&1 | grep -v "missing"
done

echo "Generating XeISA"
for file in *.gen; do
  echo "Processing $file"
  kernel_name=$(grep '^\.kernel' $file | awk -F'[@()]' '{print $2}')
  grep -v "Inline assembly" $file > ${kernel_name}.pisa

  llc -march=xe -swsb-allocation ${kernel_name}.pisa
  llc -march=xe -swsb-allocation -filetype=obj ${kernel_name}.pisa
  elfextract ${kernel_name}.pisa.o
  mv ${kernel_name}.pisa.${kernel_name}.bin ${kernel_name}.bin
done

###################### [3] Build Executable ######################

$DPCPP_COMPILER -fsycl -fsycl-targets=spir64_gen -DVC_WA -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -Xs "-device fcs" $INCLUDE_PATHS $EXTRA_DEFS ../bgemm.cpp -o bgemm

###################### [4] Run the Executable ######################

export L0SIM_DEVICE_KIND=Xe4
export L0SIM_GRITS_PATH=${FULSIM_PATH:-/root/FCS}

if [[ -z "$L0SIM_SELECT_DEVICES" ]]; then
  export L0SIM_SELECT_DEVICES=GRITS
fi

if [[ -z "$ZESIM_PATH" ]]; then
  cd $XE4_TEST_PATH; cmake --preset debug-xe4; cd -
  export ZESIM_PATH=$XE4_TEST_PATH/build/debug-xe4/_deps/zesim-src/debug-xe4/zesim
fi

export LD_LIBRARY_PATH=$ZESIM_PATH:$LD_LIBRARY_PATH
export XE4_KERNEL_BINARY_REPLACE_PATH=./
export XE4_LOG_ON="1"                     # set to 0 to turn log off, default off
export XE4_LOG_FOLDER_PATH="./logdump"    # default log folder "dump"

./bgemm

echo "ZESIM_PATH: $ZESIM_PATH"
echo "L0SIM_SELECT_DEVICES: $L0SIM_SELECT_DEVICES"
