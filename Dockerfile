FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

WORKDIR /workspace

RUN DEBIAN_FRONTEND=noninteractive apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install nano git tmux vim -y

RUN DEBIAN_FRONTEND=noninteractive apt-get -y install python3
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install python3-pip

RUN pip install xformers
RUN pip install torchvision torchaudio

RUN pip install accelerate transformers safetensors diffusers tokenizers huggingface

RUN pip install json5

RUN pip install tqdm tomesd pytorch_lightning tensorboard
RUN pip install lion_pytorch fairscale

RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /src
RUN git clone https://github.com/timdettmers/bitsandbytes.git
WORKDIR /src/bitsandbytes
RUN CUDA_VERSION=118 make cuda11x
RUN python setup.py install

WORKDIR /workspace
COPY . .

CMD tail -F anything
