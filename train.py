from DreamlikeTrainer import DreamlikeTrainer, DreamlikeTrainerConfig
from Reporter import ReporterConfig
from Imagen import ImagenConfig
import utils

import json5
import argparse
import os

def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--project_dir', type=str, required=True, help='Path to the project directory. Must contain config.json5 file containing training parameters.')
  return parser.parse_args()


def main():
  args = get_args()
  print('Loading project ' + args.project_dir, flush=True)

  config_path = os.path.join(args.project_dir, 'config.json5')

  print('Loading config from ' + config_path, flush=True)
  with open(config_path, 'r') as file:
    json_config = json5.loads(file.read())

  print('', flush=True)
  print('Trainer config: ', flush=True)
  print(json5.dumps(json_config['trainer'], indent=2), flush=True)

  print('', flush=True)
  print('Reporter config: ', flush=True)
  print(json5.dumps(json_config['reporter'], indent=2), flush=True)
  print('', flush=True)

  print('', flush=True)
  print('Imagen config: ', flush=True)
  print(json5.dumps(json_config['imagen'], indent=2), flush=True)
  print('', flush=True)

  trainer_config = DreamlikeTrainerConfig(**json_config['trainer'], project_dir=args.project_dir)
  reporter_config = ReporterConfig(**json_config['reporter'])
  imagen_config = ImagenConfig(**json_config['imagen'])

  with utils.Timer('Initializing trainer'):
    trainer = DreamlikeTrainer(trainer_config, reporter_config, imagen_config)

  print('', flush=True)

  with utils.Timer('Training'):
    trainer.train()


if __name__ == '__main__':
  main()