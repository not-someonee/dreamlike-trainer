import torch

from DreamlikeTrainer import DreamlikeTrainer, DreamlikeTrainerConfig
from Reporter import ReporterConfig
from Imagen import ImagenConfig
from Saver import SaverConfig
import utils

import json5
import argparse
import os
from datetime import datetime

def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--config_path', type=str, required=True)
  return parser.parse_args()


args = get_args()

print('Loading config from ' + args.config_path, flush=True)
with open(args.config_path, 'r') as file:
  json_config = json5.loads(file.read())

print('', flush=True)
print('Trainer config: ', flush=True)
print(json5.dumps(json_config['trainer'], indent=2), flush=True)

print('', flush=True)
print('Saver config: ', flush=True)
print(json5.dumps(json_config['saver'], indent=2), flush=True)
print('', flush=True)

print('', flush=True)
print('Reporter config: ', flush=True)
print(json5.dumps(json_config['reporter'], indent=2), flush=True)
print('', flush=True)

print('', flush=True)
print('Imagen config: ', flush=True)
print(json5.dumps(json_config['imagen'], indent=2), flush=True)
print('', flush=True)

if 'run_name' not in json_config['trainer']:
  json_config['trainer']['run_name'] = os.path.basename(args.config_path).rsplit('.', 1)[0]

json_config['trainer']['run_name'] += '__' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

trainer_config = DreamlikeTrainerConfig(**json_config['trainer'], config_path=args.config_path)
reporter_config = ReporterConfig(**json_config['reporter'])
imagen_config = ImagenConfig(**json_config['imagen'])
saver_config = SaverConfig(**json_config['saver'])

#torch._dynamo.config.verbose=True

with utils.Timer('Initializing trainer'):
  trainer = DreamlikeTrainer(trainer_config, reporter_config, imagen_config, saver_config)

print('', flush=True)

with utils.Timer('Training'):
  trainer.train()