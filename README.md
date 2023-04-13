# Dreamlike Trainer

Opinionated Stable Diffusion trainer made for large-scale finetunes (Thousands, millions of images).  

Designed for ease of use,

## Features

- Works with multiple GPUs
- Supports tensorboard and updates in Telegram/Discord
- Supports SD 1.x and SD 2.x training
- Supports offset noise
- Supports aspect ratio bucketing
- Incremental dataset caching, allowing to start training with huge datasets right away
- Safetensor only, meaning superfast start times and fast iterations
- Based on PyTorch 2.0, with better performance compared to PyTorch 1.x
- Based on diffusers
- Easy to use and understand code
- Python API and a WebSockets API for easy integration to your services

## Installation

Dreamlike Trainer is only available as a Docker container to eliminate environment incompatabilities and make it easy to run on GPU rental platforms that support Docker (runpod.io, vast.ai, etc.)  
Dreamlike Trainer works on Windows and Linux with Nvidia GPUs with 24GB+ of VRAM. 

#### Windows
- Clone the repository 
- Install WSL2 and Docker Desktop (TODO: add more details)
- Clone 

#### Linux

## Starting the Docker container

#### Windows
Run `./run.cmd --run_path ./runs/test` in the console
#### Linux
Run `bash run.sh --run_path ./runs/test` in the console

**IMPORTANT:** Do NOT close the console, as it will kill the trainer.

## Development

- Run `python docker_copy.py && bash run.sh --dev --project_dir ./projects/test` to build the container from source, start the container, and run `python train.py` inside of it
- Run `./run.cmd --run_path ./runs/test` to skip building the container and running `docker-compose up`. This will only start the `train.py` script
- Arguments after 
- Run `python docker_copy.py` to copy source code from your machine to the docker container