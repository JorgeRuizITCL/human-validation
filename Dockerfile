FROM nvcr.io/nvidia/l4t-ml:r32.4.4-py3 as base

COPY . /lib

RUN pip install /lib
