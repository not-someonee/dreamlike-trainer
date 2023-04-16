# Dreamlike Trainer

Stable Diffusion trainer made for large-scale finetunes (Thousands, millions of images).  

## Features

- Supports tensorboard
- Supports SD 1.x and SD 2.x training
- Supports offset noise
- Supports aspect ratio bucketing
- Incremental dataset caching
- Supports safetensor for superfast start times and fast iterations
- Based on PyTorch 2.0, with better performance compared to PyTorch 1.x
- Based on diffusers
- Easy to use and understand code
- Python API and a WebSockets API for easy integration to your services

## Installation

Dreamlike Trainer is only available as a Docker container to eliminate environment incompatabilities and make it easy to run on GPU rental platforms that support Docker (runpod.io, vast.ai, etc.)  
Dreamlike Trainer works on Windows and Linux with Nvidia GPUs with 24GB+ of VRAM. 

#### Windows


#### Linux

## Starting the Docker container

#### Windows
Run `./run.bat --config_path ./runs/test` in the console

**IMPORTANT:** Do NOT close the console, as it will kill the trainer.

## Development

- Run `python docker_copy.py` to copy source code from your machine to the docker container
- Run `python docker_copy.py && bash run.sh --build-up-dev --config_path ./projects/test` to build the container from source, start the container, and start `train.py` inside of it
- Run `python docker_copy.py && bash run.sh --dev --config_path ./projects/test` to skip building the container and running `docker-compose up`. This will only start the `train.py` script