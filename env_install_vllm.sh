#!/bin/bash

# Memory management
# export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export HSA_ENABLE_SDMA=0 

# Set environment variables
export VLLM_TARGET_DEVICE=rocm
export ROCM_PATH=/opt/rocm
export SETUPTOOLS_SCM_PRETEND_VERSION=0.8.5.dev

# # Change to workspace directory
# cd /workspace/

# Uninstall existing vllm if present
pip uninstall -y vllm || true

# # Remove existing vllm-patch directory if exists
# rm -rf vllm-patch

# # Clone and install vllm
# git clone https://github.com/RLFoundation/vllm-patch.git
cd vllm-patch
git checkout v0.8.5-sleep-numa

# Clean up any existing build files
rm -rf build/ dist/ *.egg-info

# Create symbolic link for ROCm library
ln -sf /opt/rocm/lib/libamdhip64.so /usr/lib/libamdhip64.so

# Install vllm with specific configurations
PYTORCH_ROCM_ARCH="gfx90a;gfx942" MAX_JOBS=${MAX_JOBS} python3 setup.py install