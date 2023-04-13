import os

files = [
  'train.py',
  'DreamlikeTrainer.py',
  'RawDataset.py',
  'CachedDataset.py',
  'bucketing_utils.py',
  'Reporter.py',
  'Imagen.py',
  'utils.py',
  'saving_utils.py',
  'train_utils.py',
  'Validator.py',
  # 'accelerate_config.yml',
]
directories = ['weights']  # path/to/dir

for file in files:
  print(file)
  os.system(f'docker cp {file} dreamlike-trainer:/workspace/' + file.rsplit('/', 1)[0])

for directory in directories:
  print(directory)
  os.system(f'docker cp {directory}/. dreamlike-trainer:/workspace/{directory}')
