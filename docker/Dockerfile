FROM supervisely/base-py-sdk:6.72.30
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update

RUN pip3 install torch==1.10.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip3 install torchvision==0.11.2+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip3 install ultralytics==8.2.26

RUN python3 -m pip install git+https://github.com/supervisely/supervisely.git@tracking-algorithms#egg=supervisely[tracking]
RUN python3 -c "from supervisely.nn.tracker import *"
