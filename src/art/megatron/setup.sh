#!/usr/bin/env bash
set -euo pipefail

export CUDA_HOME="/usr/local/cuda-12.8"
export TORCH_CUDA_ARCH_LIST="9.0"
# install missing cudnn headers & ninja build tools
apt-get update
apt-get install -y libcudnn9-headers-cuda-12 ninja-build
# install apex
if [ -d /root/apex ]; then
  echo "apex directory already exists, skipping clone"
else
  git clone --depth 1 --branch 25.09 https://github.com/NVIDIA/apex.git /root/apex
fi
NVCC_APPEND_FLAGS="--threads 4" APEX_PARALLEL_BUILD=16 APEX_CPP_EXT=1 APEX_CUDA_EXT=1 APEX_FAST_LAYER_NORM=1 uv pip install --no-build-isolation /root/apex
# install flash attention
# git clone https://github.com/Dao-AILab/flash-attention.git /root/flash-attention
# (cd /root/flash-attention && git checkout 27f501d)
# uv run /root/flash-attention/hopper/setup.py install
# install transformer engine and megatron
# Build transformer-engine-torch from source with --no-build-isolation to use venv's torch headers
# (prevents ABI mismatch with system PyTorch in the container)
echo "transformer-engine>=2.11.0" > /tmp/te-override.txt
uv pip install --no-build-isolation --override /tmp/te-override.txt \
  transformer-engine==2.11.0 \
  transformer-engine-cu12==2.11.0 \
  transformer-engine-torch==2.11.0 \
  megatron-core==0.15.2 \
  megatron-bridge==0.2.0rc6
rm /tmp/te-override.txt
# silence pynvml warnings
uv pip uninstall pynvml
uv pip install nvidia-ml-py==13.580.82
