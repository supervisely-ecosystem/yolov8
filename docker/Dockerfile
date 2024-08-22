FROM supervisely/base-py-sdk:6.72.30
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update

RUN pip3 install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install ultralytics==8.2.79
RUN pip3 install protobuf==3.20.1
RUN pip3 install onnx==1.12.0

RUN python3 -m pip install supervisely[tracking]==6.73.158
RUN python3 -c "from supervisely.nn.tracker import *"
RUN pip3 install dill==0.3.8

LABEL python_sdk_version=6.73.158