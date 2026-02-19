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
    PIP_PREFER_BINARY=1

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
  && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install pip==26.0.1
RUN pip install setuptools==78.1.1 wheel==0.46.2

RUN pip install nvidia-cudnn-cu11==8.9.4.25 --no-cache-dir
RUN pip install --pre --extra-index-url https://pypi.nvidia.com/ \
  tensorrt==9.0.1.post11.dev4 --no-cache-dir

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
FROM supervisely/py-sdk-hardened:${tag_ref_name} AS runtime

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

ENV PATH="/opt/venv/bin:$PATH"

LABEL python_sdk_version="$tag_ref_name"
