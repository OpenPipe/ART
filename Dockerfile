FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel

ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV MAX_JOBS=8
ENV RL_DOCKER=1

# cuDNN headers are installed via pip at nvidia/cudnn — expose them for TE/flash-attn builds
ENV CUDNN_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cudnn
ENV CUDNN_INCLUDE_DIR=${CUDNN_PATH}/include
ENV CUDNN_LIBRARY=${CUDNN_PATH}/lib
RUN ln -sf ${CUDNN_PATH}/include/cudnn*.h /usr/local/cuda/include/ && \
    ln -sf ${CUDNN_PATH}/lib/libcudnn*.so* /usr/local/cuda/lib64/

# ninja speeds up TE/flash-attn compilation; uv for fast installs + MCP server
RUN pip install uv ninja

# megatron-core + megatron-bridge (installs TE prebuilt wheel — will be overwritten below)
RUN uv pip install --system megatron-core megatron-bridge==0.2.0rc6

# art[backend] — vllm==0.13.0 pins torch==2.9.0, exact match
COPY pyproject.toml README.md /tmp/art-pkg/
COPY src/ /tmp/art-pkg/src/
RUN uv pip install --system "/tmp/art-pkg[backend]"

# flash-attn from source (must match container torch ABI)
RUN uv pip install --system --no-build-isolation --force-reinstall --no-deps flash-attn

# Fix TE ABI: PyPI prebuilt TE wheels were compiled against NVIDIA's custom torch,
# not the PyPI torch 2.9.0. Rebuild from GitHub source so the .so matches.
RUN apt-get update && apt-get install -y --no-install-recommends git cmake && rm -rf /var/lib/apt/lists/*
RUN pip install pybind11 nvidia-mathdx
RUN pip uninstall -y transformer-engine transformer-engine-cu12 transformer-engine-torch && \
    NVTE_FRAMEWORK=pytorch pip install --no-build-isolation \
    "transformer_engine[pytorch] @ git+https://github.com/NVIDIA/TransformerEngine.git@v2.9"

# unsloth sub-deps
RUN uv pip install --system hf_transfer tyro cut_cross_entropy "datasets>=3.4.1,<4.4.0"

# Fix pynvml deprecation warning (megatron-core pulls pynvml, torch prefers nvidia-ml-py)
RUN pip uninstall -y pynvml 2>/dev/null; uv pip install --system nvidia-ml-py

# grouped_gemm for MoE LoRA (used by megatron-core MoE layers)
RUN pip install "grouped_gemm @ git+https://github.com/fanshiqing/grouped_gemm@v1.1.4"

# Fix TE triton kernel: core.get_int_dtype() not hashable by Triton 3.5.0 JIT
RUN python3 -c "\
path = '/opt/conda/lib/python3.11/site-packages/transformer_engine/pytorch/triton/permutation.py'; \
f = open(path); c = f.read(); f.close(); \
old = '    idtype = core.get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)'; \
new = '    # Inlined get_int_dtype to avoid Triton 3.5+ JIT cache_key issue\n    bw: tl.constexpr = x.dtype.primitive_bitwidth\n    if bw == 8:\n        idtype = tl.int8\n    elif bw == 16:\n        idtype = tl.int16\n    elif bw == 32:\n        idtype = tl.int32\n    else:\n        idtype = tl.int64'; \
assert old in c, 'TE patch target not found'; \
f = open(path, 'w'); f.write(c.replace(old, new)); f.close(); \
print('TE triton patch applied')"

RUN rm -rf /tmp/art-pkg
WORKDIR /workspace
