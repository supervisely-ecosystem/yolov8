# ============================================================================
# STAGE 1: Builder Stage
# ============================================================================
# todo: remove; temp
# ARG tag_ref_name=6.73.531
ARG tag_ref_name=0.0.1

# NOTE: CUDA Python wheels and NVIDIA runtime support are primarily Linux/x86_64.
# Building on Apple Silicon should use buildx/emulation; force linux/amd64 here.
FROM --platform=linux/amd64 python:3.10-slim-bookworm AS builder

ENV LANG=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_PREFER_BINARY=1 \
    DEBIAN_FRONTEND=noninteractive

# Base build + runtime shared libs
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  ca-certificates \
  ffmpeg \
  git \
  libboost-all-dev \
  libexiv2-27 \
  libgeos-c1v5 \
  libsm6 \
  libxext6 \
  libxrender1 \
  wget \
  && rm -rf /var/lib/apt/lists/*

# Install CUDA runtime libs from NVIDIA Debian 12 repo (bookworm)
# This ensures shared libs are present at OS level for CUDA-enabled wheels.
RUN set -eux; \
  wget -qO /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb; \
  dpkg -i /tmp/cuda-keyring.deb; \
  rm -f /tmp/cuda-keyring.deb; \
  apt-get update; \
  apt-get install -y --no-install-recommends \
    cuda-cudart-12-1 \
    libcublas-12-1 \
    libcufft-12-1 \
    libcurand-12-1 \
    libcusolver-12-1 \
    libcusparse-12-1; \
  rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install pip==26.0.1
RUN pip install setuptools==78.1.1 wheel==0.46.2

# Keep Python CUDA wheels too (some packages expect these package-provided libs)
RUN pip install --no-cache-dir \
  nvidia-cuda-runtime-cu12 \
  nvidia-cudnn-cu12

RUN pip install networkx==2.8.8
RUN pip install torch==2.3.1 torchvision==0.18.1 \
  --index-url https://download.pytorch.org/whl/cu121
RUN pip install protobuf==3.20.1
RUN pip install onnx==1.12.0

RUN pip install requests==2.28.2 urllib3==1.26.15
RUN pip install packaging==21.3
RUN pip install dill==0.3.8
RUN pip install ruamel.yaml==0.17.21

RUN pip install ultralytics==8.3.15
RUN pip install onnxruntime-gpu==1.18.0 \
  --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
RUN pip install dill==0.2.8.2 tf2onnx==1.16.1 skl2onnx==1.17.0

# todo: remove; temp
# RUN pip install supervisely[tracking,model-benchmark]==$tag_ref_name
# RUN python -c "from supervisely.nn.tracker import *"
RUN pip install git+https://github.com/supervisely/supervisely.git@hardened-docker-images

RUN pip install --upgrade \
  "protobuf>=6.33.5" \
  "jaraco.context>=6.1.0" \
  wheel==0.46.2 \
  && rm -rf \
  /opt/venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco.context* \
  /opt/venv/lib/python3.10/site-packages/setuptools/_vendor/wheel*

# ============================================================================
# STAGE 2: Runtime Stage (Hardened)
# ============================================================================
FROM --platform=linux/amd64 supervisely/py-sdk-hardened:${tag_ref_name} AS runtime

ARG tag_ref_name

# Copy Python environment and required system libs from builder
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /usr/bin/ffmpeg /usr/bin/ffmpeg
COPY --from=builder /usr/lib/*-linux-gnu/libav*.so.* /usr/lib/
COPY --from=builder /usr/lib/*-linux-gnu/libsw*.so.* /usr/lib/
COPY --from=builder /usr/lib/*-linux-gnu/libgeos*.so.* /usr/lib/
COPY --from=builder /usr/lib/*-linux-gnu/libSM.so.* /usr/lib/
COPY --from=builder /usr/lib/*-linux-gnu/libXext.so.* /usr/lib/
COPY --from=builder /usr/lib/*-linux-gnu/libXrender.so.* /usr/lib/
COPY --from=builder /usr/lib/*-linux-gnu/libexiv2.so.* /usr/lib/
COPY --from=builder /usr/lib/*-linux-gnu/libboost*.so.* /usr/lib/

# Copy CUDA runtime libs required by torch cu121 / onnxruntime-gpu
COPY --from=builder /usr/lib/*-linux-gnu/libcudart.so.* /usr/lib/
COPY --from=builder /usr/lib/*-linux-gnu/libcublas.so.* /usr/lib/
COPY --from=builder /usr/lib/*-linux-gnu/libcublasLt.so.* /usr/lib/
COPY --from=builder /usr/lib/*-linux-gnu/libcufft.so.* /usr/lib/
COPY --from=builder /usr/lib/*-linux-gnu/libcurand.so.* /usr/lib/
COPY --from=builder /usr/lib/*-linux-gnu/libcusolver.so.* /usr/lib/
COPY --from=builder /usr/lib/*-linux-gnu/libcusparse.so.* /usr/lib/
COPY --from=builder /usr/lib/*-linux-gnu/libnvrtc.so.* /usr/lib/
COPY --from=builder /usr/lib/*-linux-gnu/libnvJitLink.so.* /usr/lib/

# remove OS package managers & related metadata
RUN set -eux; \
  rm -rf /var/lib/apt/lists/* /etc/apt /var/cache/apt /var/lib/apt; \
  rm -f \
    /usr/bin/apt /usr/bin/apt-get /usr/bin/apt-cache /usr/bin/apt-config /usr/bin/apt-key \
    /usr/bin/dpkg /usr/bin/dpkg-query /usr/bin/dpkg-deb /usr/bin/dpkg-divert; \
  rm -rf /var/lib/dpkg /usr/share/dpkg || true

ENV PATH="/opt/venv/bin:$PATH" \
    LD_LIBRARY_PATH="/usr/lib:/usr/local/lib:${LD_LIBRARY_PATH}"

LABEL python_sdk_version="$tag_ref_name"