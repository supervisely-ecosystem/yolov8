FROM supervisely/base-py-sdk:6.72.30
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update

RUN pip3 install torch==1.10.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip3 install torchvision==0.11.2+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip3 install ultralytics==8.0.112

RUN python3 -m pip install supervisely==6.73.89
