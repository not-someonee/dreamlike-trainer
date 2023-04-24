FROM nvidia/cuda:12.1.0-devel-ubuntu20.04
WORKDIR /workspace

RUN DEBIAN_FRONTEND=noninteractive apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install nano git tmux vim htop -y

RUN DEBIAN_FRONTEND=noninteractive apt-get -y install python3
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install python3-pip

RUN pip install xformers
RUN pip install torchvision torchaudio

RUN pip install accelerate transformers safetensors diffusers tokenizers huggingface

RUN pip install json5 py-spy

RUN pip install tqdm tomesd pytorch_lightning tensorboard
RUN pip install lion_pytorch fairscale

RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /src
RUN git clone https://github.com/timdettmers/bitsandbytes.git
WORKDIR /src/bitsandbytes
RUN CUDA_VERSION=121 make cuda12x
RUN CUDA_VERSION=121 make cuda12x_nomatmul
RUN python setup.py install

WORKDIR /workspace
COPY . .

CMD tail -F anything
