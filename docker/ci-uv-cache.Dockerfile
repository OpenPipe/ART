ARG BASE_IMAGE=pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel
FROM ${BASE_IMAGE}

ARG CI_PYTHON_MM=3.11
ARG BUILD_JOBS=2

ENV UV_CACHE_DIR=/root/.cache/uv
ENV UV_LINK_MODE=copy
ENV UV_CONCURRENT_BUILDS=${BUILD_JOBS}
ENV CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS}
ENV MAX_JOBS=${BUILD_JOBS}
ENV NINJAFLAGS=-j${BUILD_JOBS}
ENV TORCH_CUDA_ARCH_LIST=8.0

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates curl git && \
    rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /opt/art-uv-cache
COPY pyproject.toml uv.lock ./

# Pre-warm uv cache with the full CI dependency surface.
RUN uv sync --frozen --all-extras --group dev --no-install-project && \
    rm -rf /opt/art-uv-cache/.venv

# Archive the cache inside the image for easy extraction.
RUN tar -C "${UV_CACHE_DIR}" -cf /tmp/uv-cache.tar .
