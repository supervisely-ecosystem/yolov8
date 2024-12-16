FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install python3-pip -y
RUN apt-get install -y git

RUN pip3 install nvidia-cudnn-cu11==8.9.4.25 --no-cache-dir
RUN pip3 install --pre --extra-index-url https://pypi.nvidia.com/ tensorrt==9.0.1.post11.dev4 --no-cache-dir

RUN pip3 install networkx==2.8.8
RUN pip3 install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install protobuf==3.20.1
RUN pip3 install onnx==1.12.0

RUN apt update && apt-get install ffmpeg libgeos-dev libsm6 libxext6 libexiv2-dev libxrender-dev libboost-all-dev -y

RUN pip3 install requests==2.28.2 urllib3==1.26.15
RUN pip3 install packaging==21.3
RUN pip3 install dill==0.3.8
RUN pip3 install ruamel.yaml==0.17.21

RUN pip3 install ultralytics==8.3.15
RUN pip3 install onnxruntime-gpu==1.18.0
RUN pip3 install dill==0.2.8.2 tf2onnx==1.16.1 skl2onnx==1.17.0
RUN python3 -m pip install supervisely[tracking,model-benchmark]==6.73.256
RUN python3 -c "from supervisely.nn.tracker import *"

LABEL python_sdk_version=6.73.256
