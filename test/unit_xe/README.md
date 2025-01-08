
# Introduction

Welcome to the guide on building Xe4 unit testing that generates pISA assembly code. This document provides a step-by-step process to help you set up your environment and build your testing code.

# Set Up the Docker Container

To streamline the development process, we've prepared a Docker image that bundles all the necessary tools and libraries. To set up your Docker container, please follow the instructions provided in [the sim_docker manual](https://github.com/intel-sandbox/containers.docker.gpu.compute.sim?tab=readme-ov-file#overview).

# Inside the Docker Container

## Install CUDA Toolkit

```bash
# (optional) export http_proxy if wget timeout
export http_proxy=http://child-prc.intel.com:913
export https_proxy=http://child-prc.intel.com:913
env | grep proxy

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt update && apt -y install cuda-toolkit-12-4
```

## Git Clone and Submodule

```bash
git clone https://github.com/intel-sandbox/libraries.gpu.xe.cutlass_pisa.git
cd libraries.gpu.xe.cutlass_pisa
git submodule update --init
```

## Upgrade XeSim/Compiler/ZeSim

```bash
git submodule update --init
./test/unit_xe/3rdparty/drivers.gpu.compute.workloads/simt_workloads/scripts
```

## Functional Test

* Bgemm

Before execution, double-check `/root/zesim/debug/zesim/zesim.yml`, make sure `-enableFeature clusterSupportForSystolic2` not in variable `L0SIM_GRITS_AUBLOAD_OPTS` defined at line 16.

```bash
cd ./test/unit_xe/functional/bgemm
./run_online.sh
```

* Cluster Bgemm

Before execution, modify `/root/zesim/debug/zesim/zesim.yml`, append option `-enableFeature clusterSupportForSystolic2` at line 13 to variable `L0SIM_GRITS_AUBLOAD_OPTS` defined at line 16.

```bash
cd ./test/unit_xe/functional/cluster_bgemm
./run_online.sh
```

## Perf Test

* Bgemm

Before execution, modify `/root/zesim/debug/zesim/zesim.yml`, append option `-attr MEMPIPE_DMA_READ_BW 2 -wave_dump timed -cb_cfg waveform_trace_stats true waveform_trace_timegraph true waveform_tg_cfg_filename TG_ArchTarget_xe4.txt waveform_tg_interval 100` at line 12 to variable `L0SIM_GRITS_AUBLOAD_OPTS` defined at line 16.

```bash
cd ./test/unit_xe/functional/bgemm
./run_online.sh
```

* Cluster Bgemm

Before execution, modify `/root/zesim/debug/zesim/zesim.yml`, append option `-attr MEMPIPE_DMA_READ_BW 2 -wave_dump timed -cb_cfg waveform_trace_stats true waveform_trace_timegraph true waveform_tg_cfg_filename TG_ArchTarget_xe4.txt waveform_tg_interval 100` at line 12 and option `-enableFeature clusterSupportForSystolic2` at line 13 to variable `L0SIM_GRITS_AUBLOAD_OPTS` defined at line 16.

```bash
cd ./test/unit_xe/functional/cluster_bgemm
./run_online.sh
```
