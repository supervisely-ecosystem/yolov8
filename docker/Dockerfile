# ============================================================================
# STAGE 1: Builder Stage
# ============================================================================
# todo: remove; temp
# ARG tag_ref_name=6.73.531
ARG tag_ref_name=0.0.1

FROM python:3.10-slim-bookworm AS builder

ENV LANG=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_PREFER_BINARY=1 \
    DEBIAN_FRONTEND=noninteractive

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

# Install NVIDIA CUDA repo and CUDA runtime/toolkit components
RUN set -eux; \
  wget -qO /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb; \
  dpkg -i /tmp/cuda-keyring.deb; \
  rm -f /tmp/cuda-keyring.deb; \
  apt-get update; \
  apt-get install -y --no-install-recommends \
    cuda-toolkit-12-6 \
    cuda-cudart-12-6 \
    libcublas-12-6 \
    libcufft-12-6 \
    libcurand-12-6 \
    libcusolver-12-6 \
    libcusparse-12-6; \
  rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install pip==26.0.1 setuptools==78.1.1 wheel==0.46.2

# Keep NumPy on 1.x for binary wheels that are not yet NumPy 2 compatible
RUN pip install --no-cache-dir numpy==1.26.4

RUN pip install networkx==2.8.8
RUN pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
RUN pip install protobuf==3.20.1 onnx==1.12.0
RUN pip install requests==2.28.2 urllib3==1.26.15 packaging==21.3 ruamel.yaml==0.17.21
RUN pip install dill==0.2.8.2 tf2onnx==1.16.1 skl2onnx==1.17.0
RUN pip install ultralytics==8.3.15
RUN pip install onnxruntime-gpu==1.18.0 \
  --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

# todo: remove; temp
# RUN pip install supervisely[tracking,model-benchmark]==$tag_ref_name
# RUN python -c "from supervisely.nn.tracker import *"
RUN pip install git+https://github.com/supervisely/supervisely.git@hardened-docker-images

# ============================================================================
# STAGE 2: Runtime Stage (Hardened)
# ============================================================================
FROM supervisely/py-sdk-hardened:${tag_ref_name} AS runtime
ARG tag_ref_name

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

# CUDA runtime libs from apt-installed toolkit/runtime
COPY --from=builder /usr/local/cuda-12.6/targets/x86_64-linux/lib/libcudart.so* /usr/lib/
COPY --from=builder /usr/local/cuda-12.6/targets/x86_64-linux/lib/libcublas.so* /usr/lib/
COPY --from=builder /usr/local/cuda-12.6/targets/x86_64-linux/lib/libcublasLt.so* /usr/lib/
COPY --from=builder /usr/local/cuda-12.6/targets/x86_64-linux/lib/libcufft.so* /usr/lib/
COPY --from=builder /usr/local/cuda-12.6/targets/x86_64-linux/lib/libcurand.so* /usr/lib/
COPY --from=builder /usr/local/cuda-12.6/targets/x86_64-linux/lib/libcusolver.so* /usr/lib/
COPY --from=builder /usr/local/cuda-12.6/targets/x86_64-linux/lib/libcusparse.so* /usr/lib/
COPY --from=builder /usr/local/cuda-12.6/targets/x86_64-linux/lib/libnvrtc.so* /usr/lib/
COPY --from=builder /usr/local/cuda-12.6/targets/x86_64-linux/lib/libnvJitLink.so* /usr/lib/

RUN set -eux; \
  rm -rf /var/lib/apt/lists/* /etc/apt /var/cache/apt /var/lib/apt; \
  rm -f /usr/bin/apt /usr/bin/apt-get /usr/bin/apt-cache /usr/bin/apt-config /usr/bin/apt-key \
        /usr/bin/dpkg /usr/bin/dpkg-query /usr/bin/dpkg-deb /usr/bin/dpkg-divert; \
  rm -rf /var/lib/dpkg /usr/share/dpkg || true

ENV PATH="/opt/venv/bin:$PATH" \
    LD_LIBRARY_PATH="/usr/lib:/usr/local/lib"

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

LABEL python_sdk_version="$tag_ref_name"