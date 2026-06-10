ARG BASE_IMAGE=docker.io/pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel
ARG ART_SHA=unknown
ARG UV_VERSION=0.11.7
ARG BUILD_JOBS=2
ARG UV_CONCURRENT_BUILDS=1
ARG APEX_PARALLEL_BUILD=2
ARG APEX_NVCC_THREADS=1
ARG TORCH_CUDA_ARCH_LIST=9.0
ARG CUDNN_PACKAGE_VERSION=9.10.2.21
ARG SKYPILOT_VERSION=0.12.0
ARG SKY_REMOTE_RAY_VERSION=2.9.3

FROM ${BASE_IMAGE} AS builder

ARG UV_VERSION
ARG BUILD_JOBS
ARG UV_CONCURRENT_BUILDS
ARG APEX_PARALLEL_BUILD
ARG APEX_NVCC_THREADS
ARG TORCH_CUDA_ARCH_LIST
ARG CUDNN_PACKAGE_VERSION

ENV CUDA_HOME=/usr/local/cuda-12.8 \
    PATH=/opt/conda/bin:${PATH} \
    UV_CACHE_DIR=/opt/uv-cache \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_LINK_MODE=copy \
    UV_CONCURRENT_BUILDS=${UV_CONCURRENT_BUILDS} \
    APEX_PARALLEL_BUILD=${APEX_PARALLEL_BUILD} \
    NVCC_APPEND_FLAGS=--threads\ ${APEX_NVCC_THREADS} \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS} \
    MAX_JOBS=${BUILD_JOBS} \
    NINJAFLAGS=-j${BUILD_JOBS} \
    PYTHONUNBUFFERED=1

SHELL ["/bin/bash", "-c"]

RUN if ! getent group messagebus >/dev/null; then groupadd -r messagebus; fi \
 && dpkg-statoverride --remove /usr/lib/dbus-1.0/dbus-daemon-launch-helper || true \
 && apt-get update \
 && apt-get install -y --no-install-recommends git libibverbs-dev \
 && rm -rf /var/lib/apt/lists/* \
 && /opt/conda/bin/python -m pip install --no-cache-dir --upgrade "uv==${UV_VERSION}" \
 && /opt/conda/bin/conda clean -afy \
 && /opt/conda/bin/uv --version \
 && mkdir -p "${UV_CACHE_DIR}" "${UV_PYTHON_INSTALL_DIR}"

WORKDIR /opt/src/art
COPY pyproject.toml uv.lock ./
COPY vllm_runtime/pyproject.toml vllm_runtime/uv.lock ./vllm_runtime/

RUN /opt/conda/bin/python -m pip install --no-cache-dir "nvidia-cudnn-cu12==${CUDNN_PACKAGE_VERSION}" \
 && mkdir -p /usr/local/cuda-12.8/include /usr/local/cuda-12.8/lib64 \
 && : > /tmp/art-cudnn-symlinks.txt \
 && for src in /opt/conda/lib/python3.11/site-packages/nvidia/cudnn/include/*; do \
      dst="/usr/local/cuda-12.8/include/$(basename "$src")"; \
      if [ ! -e "$dst" ]; then ln -s "$src" "$dst" && printf '%s\n' "$dst" >> /tmp/art-cudnn-symlinks.txt; fi; \
    done \
 && for src in /opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib/*; do \
      dst="/usr/local/cuda-12.8/lib64/$(basename "$src")"; \
      if [ ! -e "$dst" ]; then ln -s "$src" "$dst" && printf '%s\n' "$dst" >> /tmp/art-cudnn-symlinks.txt; fi; \
    done \
 && UV_LINK_MODE=hardlink uv sync --frozen --extra backend --extra megatron --extra tinker --no-install-project --python 3.12 \
 && rm -rf .venv \
 && cd vllm_runtime \
 && UV_LINK_MODE=hardlink uv sync --frozen --no-install-project --no-dev --python 3.12 \
 && rm -rf .venv \
 && if [ -f /tmp/art-cudnn-symlinks.txt ]; then while IFS= read -r link; do [ -L "$link" ] && rm "$link"; done < /tmp/art-cudnn-symlinks.txt; fi \
 && rm -f /tmp/art-cudnn-symlinks.txt

FROM ${BASE_IMAGE}

ARG ART_SHA
ARG UV_VERSION
ARG BUILD_JOBS
ARG UV_CONCURRENT_BUILDS
ARG APEX_PARALLEL_BUILD
ARG APEX_NVCC_THREADS
ARG TORCH_CUDA_ARCH_LIST
ARG SKYPILOT_VERSION
ARG SKY_REMOTE_RAY_VERSION

ENV CUDA_HOME=/usr/local/cuda-12.8 \
    PATH=/home/sky/.local/bin:/opt/conda/bin:${PATH} \
    UV_CACHE_DIR=/opt/uv-cache \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_LINK_MODE=copy \
    UV_CONCURRENT_BUILDS=${UV_CONCURRENT_BUILDS} \
    APEX_PARALLEL_BUILD=${APEX_PARALLEL_BUILD} \
    NVCC_APPEND_FLAGS=--threads\ ${APEX_NVCC_THREADS} \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS} \
    MAX_JOBS=${BUILD_JOBS} \
    NINJAFLAGS=-j${BUILD_JOBS} \
    PYTHONUNBUFFERED=1 \
    HOME=/home/sky

SHELL ["/bin/bash", "-c"]

LABEL org.opencontainers.image.source="https://github.com/openpipe/art" \
      org.opencontainers.image.revision="${ART_SHA}" \
      org.opencontainers.image.description="ART GPU image with warmed uv caches for SkyPilot launches." \
      org.opencontainers.image.title="art-gpu"

# Keep apt metadata available because SkyPilot/setup may still run apt-get in
# fresh clusters, and preinstall the small tools ART's setup expects.
RUN if ! getent group messagebus >/dev/null; then groupadd -r messagebus; fi \
 && dpkg-statoverride --remove /usr/lib/dbus-1.0/dbus-daemon-launch-helper || true \
 && apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      -o Dpkg::Options::=--force-confdef \
      -o Dpkg::Options::=--force-confold \
      curl \
      fuse \
      gcc \
      git \
      htop \
      jq \
      libcudnn9-headers-cuda-12 \
      libibverbs-dev \
      nano \
      netcat-openbsd \
      ninja-build \
      nvtop \
      openssh-server \
      patch \
      pciutils \
      rsync \
      socat \
      sudo \
      tmux \
      unzip \
      wget \
 && mkdir -p /var/run/sshd "${UV_CACHE_DIR}" "${UV_PYTHON_INSTALL_DIR}" \
 && sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config \
 && sed -i 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' /etc/pam.d/sshd \
 && ssh-keygen -A \
 && useradd -m -s /bin/bash sky \
 && mkdir -p /home/sky/.local/bin /home/sky/.sky/sky_app \
 && /bin/bash -c 'echo "sky ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers' \
 && /bin/bash -c 'echo '\''Defaults secure_path="/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"'\'' > /etc/sudoers.d/sky' \
 && /opt/conda/bin/python -m pip install --no-cache-dir --upgrade "uv==${UV_VERSION}" \
 && /opt/conda/bin/conda clean -afy \
 && ln -sf /opt/conda/bin/uv /home/sky/.local/bin/uv \
 && /opt/conda/bin/uv --version \
 && chown -R sky:sky /home/sky "${UV_CACHE_DIR}" "${UV_PYTHON_INSTALL_DIR}"

COPY --from=builder --chown=sky:sky /opt/uv-cache /opt/uv-cache
COPY --from=builder --chown=sky:sky /opt/uv-python /opt/uv-python

USER sky
WORKDIR /home/sky

RUN mkdir -p "${HOME}/.local/bin" "${HOME}/.sky/sky_app" "${HOME}/sky_workdir" \
 && ln -sf /opt/conda/bin/uv "${HOME}/.local/bin/uv" \
 && uv venv --seed "${HOME}/skypilot-runtime" --python 3.10 \
 && VIRTUAL_ENV="${HOME}/skypilot-runtime" UV_LINK_MODE=copy UV_SYSTEM_PYTHON=false env -u PYTHONPATH -C "${HOME}" uv pip install \
      "setuptools<70" \
      "skypilot[kubernetes,remote]==${SKYPILOT_VERSION}" \
      "ray[default]==${SKY_REMOTE_RAY_VERSION}" \
      "pycryptodome==3.12.0" \
 && VIRTUAL_ENV="${HOME}/skypilot-runtime" UV_LINK_MODE=copy UV_SYSTEM_PYTHON=false env -u PYTHONPATH -C "${HOME}" uv pip uninstall skypilot \
 && printf '%s\n' "${HOME}/skypilot-runtime/bin/python" > "${HOME}/.sky/python_path" \
 && VIRTUAL_ENV="${HOME}/skypilot-runtime" UV_LINK_MODE=copy UV_SYSTEM_PYTHON=false env -u PYTHONPATH -C "${HOME}" uv run --no-project --no-config which ray > "${HOME}/.sky/ray_path"
