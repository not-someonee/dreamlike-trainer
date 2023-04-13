@echo off
setlocal enabledelayedexpansion

set "args="

if "%1" == "--build-dev" (
    docker build -t dreamlike-trainer .
) else if "%1" == "--up-dev" (
    docker-compose -f docker-compose.dev.yml up -d
) else if "%1" == "--build-up-dev" (
    docker build -t dreamlike-trainer .
    docker-compose -f docker-compose.dev.yml up -d
) else if not "%1" == "--dev" (
    docker-compose -f docker-compose.yml up -d
)

:argloop
if "%1" == "" goto endargs
if "%1" NEQ "--dev" (
    if "%1" NEQ "--build-dev" (
        if "%1" NEQ "--up-dev" (
            if "%1" NEQ "--build-up-dev" (
                set "args=!args! %1"
            )
        )
    )
)
shift
goto argloop

:endargs
docker exec -it dreamlike-trainer bash -c "accelerate launch --config_file ./accelerate_config.yml train.py !args!"