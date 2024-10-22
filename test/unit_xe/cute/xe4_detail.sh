#!/usr/bin/env bash

set -e

source /opt/intel/oneapi/setvars.sh

SCRIPT_PATH=$(dirname "$(realpath "$0")")
XE4_TEST_PATH=$(realpath "$SCRIPT_PATH/../3rdparty/xe4_test")

icpx -fsycl xe4_detail.cpp -I ../../../include/ -I$XE4_TEST_PATH/pisa_tests -I /usr/local/cuda/include -o xe4_detail
./xe4_detail
