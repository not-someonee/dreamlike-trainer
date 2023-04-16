import os

files = [
  # 'setup_env.py',
  'train.py',
  'Validator.py',
  'Controller.py',
  'train_utils.py',
  # 'sd_utils.py',
  # 'keyboard_util.py',
  'SDPipeline.py',
  # 'saving_utils.py',
  'DreamlikeTrainer.py',
  'RawDataset.py',
  # 'CachedDataset.py',
  # 'bucketing_utils.py',
  'Reporter.py',
  'Imagen.py',
  # 'bcolors.py',
  'Saver.py',
  # 'utils.py',
  # 'saving_utils.py',
  # 'train_utils.py',
  'accelerate_config.yml',
]
directories = []  # path/to/dir

for file in files:
  print(file)
  os.system(f'docker cp {file} dreamlike-trainer:/workspace/' + file.rsplit('/', 1)[0])

for directory in directories:
  print(directory)
  os.system(f'docker cp {directory}/. dreamlike-trainer:/workspace/{directory}')
