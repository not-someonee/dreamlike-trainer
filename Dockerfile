FROM nvidia/cuda:11.7.0-devel-ubuntu22.04

WORKDIR /workspace

RUN DEBIAN_FRONTEND=noninteractive apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install nano git -y

RUN DEBIAN_FRONTEND=noninteractive apt-get -y install python3
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install python3-pip

RUN pip install xformers
RUN pip install torchvision torchaudio

RUN pip install accelerate transformers safetensors diffusers tokenizers huggingface

RUN pip install bitsandbytes-cuda117 json5

RUN pip install tqdm tomesd pytorch_lightning tensorboard
RUN pip install lion_pytorch

RUN ln -s /usr/bin/python3 /usr/bin/python

COPY . .

CMD tail -F anything
