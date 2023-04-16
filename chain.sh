#!/bin/bash

./run.bat --dev --config_path ./configs/te_lr_1.5x_cosine.json5
./run.bat --dev --config_path ./configs/te_lr_1x_constant.json5
./run.bat --dev --config_path ./configs/te_lr_0.5x_constant.json5
./run.bat --dev --config_path ./configs/te_lr_0.25x_constant.json5

