FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

ARG DEBIAN_FRONTEND=noninteractive

RUN apt -y update && \
    apt install -y git build-essential ssh python3-tk

COPY requirement.txt /tmp
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirement.txt
