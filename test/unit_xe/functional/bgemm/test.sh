#!/usr/bin/env bash

source /opt/intel/oneapi/setvars.sh

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
XE4_INCLUDE_PATH=$(realpath "$SCRIPT_PATH/../../../3rdparty/xe4_test/pisa_tests")

INCLUDE_PATHS="-I/root/working_dir/libraries.gpu.xe.cutlass_pisa/include \
               -I/usr/local/cuda/include \
               -I$XE4_INCLUDE_PATH \
               -I/root/working_dir/libraries.gpu.xe.cutlass_pisa/test/unit_xe/functional"

$DPCPP_COMPILER -fsycl -fsycl-targets=spir64_gen -Xs "-device fcs" -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -std=c++17 -fPIE $INCLUDE_PATHS $EXTRA_DEFS ../bgemm.cpp -o bgemm

for file in igc_dump/*.spv; do
  echo "Processing $file"
  LD_LIBRARY_PATH=$IGC_20_PATH/lib:$LD_LIBRARY_PATH \
  ocloc compile -device fcs -spirv_input -gen_file -output_no_suffix -file $file 2>&1 | grep -v "missing"
done

rm *.bin
kernel_name=_ZTS5BGEMM

echo "Generating XeISA"
for file in *.gen; do
  echo "Processing $file"
  grep -v "Inline assembly" $file > ${kernel_name}.pisa

  llc -march=xe -swsb-allocation ${kernel_name}.pisa
  llc -march=xe -swsb-allocation -filetype=obj ${kernel_name}.pisa
  elfextract ${kernel_name}.pisa.o
  mv ${kernel_name}.pisa.${kernel_name}.bin ${kernel_name}.bin
done

$DPCPP_COMPILER -fsycl -fsycl-targets=spir64_gen -DVC_WA -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -Xs "-device fcs" $INCLUDE_PATHS $EXTRA_DEFS ../bgemm.cpp -o bgemm

REPO_ROOT=/root/working_dir/drivers.gpu.simulation.gen-isa-interpreter

export LD_LIBRARY_PATH=$REPO_ROOT/build/debug-xe4/runtime/src:$LD_LIBRARY_PATH
export L0SIM_DEVICE_KIND=Xe4
export L0SIM_GRITS_PATH=/root/FCS

# export L0SIM_SELECT_DEVICES=XE4ISAI
export L0SIM_SELECT_DEVICES=GRITS

export XE4_KERNEL_BINARY_REPLACE_PATH=./
export XE4_LOG_ON="1"                     # set to 0 to turn log off, default off
export XE4_LOG_FOLDER_PATH="./logdump"    # default log folder "dump"

./bgemm
