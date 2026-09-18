# syntax=docker/dockerfile:1
FROM modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.6.3-py311-torch2.7.1-vllm0.10.0-modelscope1.28.2-swift3.7.2

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Core packages SkyPilot expects on the instance, plus distutils for gpustat build
RUN apt-get update && apt-get install -y --no-install-recommends \
        openssh-server \
        rsync \
        netcat-openbsd \
        pciutils \
        libpci3 \
        fuse3 \
        libfuse3-3 \
        libfuse2 \
        python3.10 \
        python3-pip \
        python3.10-venv \
        python3-distutils \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Ensure SSH server can start (SkyPilot uses SSH inside pods)
RUN mkdir -p /var/run/sshd && \
    sed -i 's/#\?PermitRootLogin .*/PermitRootLogin yes/' /etc/ssh/sshd_config || true

# Preinstall Ray and SkyPilot deps into system Python 3.10 so /usr/local/bin/ray works
RUN /usr/bin/python3.10 -m pip install --upgrade "pip<25.1" "setuptools<70" && \
    /usr/bin/python3.10 -m pip install --no-cache-dir "gpustat==1.1.1" "ray[default]==2.9.3" "skypilot==0.10.3"

# Also prepare the SkyPilot runtime venv to skip runtime setup where possible
RUN python3.10 -m venv /root/skypilot-runtime && \
    . /root/skypilot-runtime/bin/activate && \
    pip install --upgrade "pip<25.1" "setuptools<70" && \
    pip install --no-cache-dir "gpustat==1.1.1" "ray[default]==2.9.3" "skypilot==0.10.3"

# Keep base image entrypoint/cmd


