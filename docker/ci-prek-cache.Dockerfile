ARG BASE_IMAGE=pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel
FROM ${BASE_IMAGE}

ARG CI_PYTHON_MM=3.11
ARG MEGATRON_FINGERPRINT=unset
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

WORKDIR /opt/art-prek-bootstrap
COPY pyproject.toml uv.lock ./

# Pre-warm uv cache with the full CI dependency surface while avoiding editable install.
RUN uv sync --frozen --all-extras --group dev --no-install-project && \
    rm -rf /opt/art-prek-bootstrap/.venv

RUN mkdir -p /etc/art-ci && \
    printf '%s' "${MEGATRON_FINGERPRINT}" > /etc/art-ci/megatron_fingerprint && \
    printf '%s' "${BASE_IMAGE}" > /etc/art-ci/base_image && \
    printf '%s' "${CI_PYTHON_MM}" > /etc/art-ci/python_mm

WORKDIR /workspace
