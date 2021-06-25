# docker build -t fishie_classification .
# docker run --rm -e "DISPLAY=${DISPLAY}" --ipc=host -it --gpus all -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" fishie_classification

FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-devel

WORKDIR /workspace
COPY ./requirements.txt /workspace/
RUN python -m pip install -r /workspace/requirements.txt

# local timm
COPY ./ /workspace/
RUN python -m pip install /workspace/

# do not use this (inplace abn is not related)
RUN python -m pip install /workspace/inplace_abn
RUN python -m pip install -r /workspace/inplace_abn/scripts/requirements.txt

ENV TZ=Asia/Singapore
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

COPY ./ /workspace
WORKDIR /workspace
