FROM supervisely/base-py-sdk:6.72.30
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update

RUN pip3 install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install ultralytics==8.2.79
RUN pip3 install protobuf==3.20.1
RUN pip3 install onnx==1.12.0

# Expected environment after installation:
# CUDA 12.1
# cuDNN 8.9.2
# TensorRT 10.3.0
RUN pip uninstall onnxruntime -y
RUN pip install cmake onnxsim==0.4.36 onnx==1.12.0 onnxruntime-gpu==1.18.0
RUN pip install nvidia-tensorrt

RUN python3 -m pip install supervisely[tracking]==6.73.203
RUN python3 -c "from supervisely.nn.tracker import *"
RUN python3 -m pip install supervisely[model-benchmark]==6.73.203
RUN pip3 install dill==0.3.8

LABEL python_sdk_version=6.73.203
