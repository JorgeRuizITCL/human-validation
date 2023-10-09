FROM nvcr.io/nvidia/l4t-ml:r32.4.4-py3 as base

# Update PIP to the latest version (compatible with the img)
RUN python3 -m pip install --upgrade pip


COPY . /lib


RUN python3 -m pip install /lib

RUN python3 -m pip install https://nvidia.box.com/shared/static/w3dezb26wog78rwm2yf2yhh578r5l144.whl

RUN wget https://nvidia.box.com/shared/static/jy7nqva7l88mq9i8bw3g3sklzf4kccn2.whl -O onnxruntime_gpu-1.10.0-cp36-cp36m-linux_aarch64.whl

RUN python3 -m pip install onnxruntime_gpu-1.10.0-cp36-cp36m-linux_aarch64.whl