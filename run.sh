#!/bin/bash

args=()

if [[ "$1" == "--build-dev" ]]; then
    docker build -t dreamlike-trainer .
elif [[ "$1" == "--up-dev" ]]; then
    docker-compose -f docker-compose.dev.yml up -d
elif [[ "$1" == "--build-up-dev" ]]; then
    docker build -t dreamlike-trainer .
    docker-compose -f docker-compose.dev.yml up -d
elif [[ "$1" != "--dev" ]]; then
    docker-compose -f docker-compose.yml up -d
fi

for arg in "$@"; do
    if [[ "$arg" != "--dev" && "$arg" != "--build-dev" && "$arg" != "--up-dev" && "$arg" != "--build-up-dev" ]]; then
        args+=("$arg")
    fi
done

docker exec -it dreamlike-trainer bash -c "accelerate launch --config_file ./accelerate_config.yml train.py ${args[*]}"
