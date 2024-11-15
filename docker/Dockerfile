FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install python3-pip -y
RUN apt-get install -y git

RUN pip3 install networkx==2.8.8
RUN pip3 install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install protobuf==3.20.1
RUN pip3 install onnx==1.12.0

RUN apt-get install ffmpeg libgeos-dev libsm6 libxext6 libexiv2-dev libxrender-dev libboost-all-dev -y

RUN pip3 install requests==2.28.2 urllib3==1.26.15
RUN pip3 install packaging==21.3
RUN pip3 install dill==0.3.8
RUN pip3 install ruamel.yaml==0.17.21

RUN pip3 install ultralytics==8.3.15
RUN python3 -m pip install supervisely[tracking,model-benchmark]==6.73.232
RUN python3 -c "from supervisely.nn.tracker import *"

LABEL python_sdk_version=6.73.232
