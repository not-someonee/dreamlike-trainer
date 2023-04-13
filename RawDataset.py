from typing import Dict, Any, List, Tuple, Literal
from dataclasses import dataclass
import os
import random
import glob
import json
from math import isclose
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms
import safetensors
import bucketing_utils
import utils


@dataclass
class RawDataItem:
  path: str
  caption: str
  width: int
  height: int
  cond_dropout: float
  shuffle_captions: bool
  native_width: int
  native_height: int

  def __post_init__(self):
    self.size = (self.width, self.height)

  def get_data(self):
    image = self.get_image()
    caption = self.get_caption()

    t = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize([0.5], [0.5]),
    ])

    return t(image), caption

  def get_caption(self):
    if not isclose(self.cond_dropout, 0.0, abs_tol=0.001):
      if random.random() < self.cond_dropout:
        return ''

    if self.shuffle_captions:
      return ', '.join([p.strip() for p in self.caption.split(',')])

    return self.caption


  # Load image from disk, ensure it has correct size
  def get_image(self, jitter=20):
    image = Image.open(self.path).convert('RGB')

    jitter_amount = random.randint(0, jitter)

    native_width = self.native_width
    native_height = self.native_height

    if self.width == self.height:
      if native_width > native_height:
        left = random.randint(0, native_width - native_height)
        image = image.crop((left, 0, native_height + left, native_height))
      elif native_height > native_width:
        top = random.randint(0, native_height - native_width)
        image = image.crop((0, top, native_width, native_width + top))
      elif native_width > self.width:
        random_trim = 0.04
        image_slice = min(int(self.width * random_trim), native_width - self.width)
        slicew_ratio = random.random()
        left = int(image_slice * slicew_ratio)
        right = native_width - int(image_slice * (1 - slicew_ratio))
        sliceh_ratio = random.random()
        top = int(image_slice * sliceh_ratio)
        bottom = native_height - int(image_slice * (1 - sliceh_ratio))

        image = image.crop((left, top, right, bottom))
    else:
      image_aspect = native_width / native_height
      target_aspect = self.width / self.height
      if image_aspect > target_aspect:
        new_width = int(native_height * target_aspect)
        jitter_amount = max(min(jitter_amount, int(abs(native_width - new_width) / 2)), 0)
        left = jitter_amount
        right = left + new_width
        image = image.crop((left, 0, right, native_height))
      else:
        new_height = int(native_width / target_aspect)
        jitter_amount = max(min(jitter_amount, int(abs(native_height - new_height) / 2)), 0)
        top = jitter_amount
        bottom = top + new_height
        image = image.crop((0, top, native_width, bottom))
    return image.resize((self.width, self.height), resample=Image.Resampling.LANCZOS)


@dataclass
class RawDatasetConfig:
  directory: str
  device: Any
  resolution: int
  batch_size: int
  cond_dropout: float
  shuffle_captions: bool
  type: Literal['train', 'val']
  seed: int = 42
  ignore_cache: bool = False
  val_split: float = 0.1
  max_val_images: int = 500



class RawDataset(Dataset):
  def __init__(self, config: RawDatasetConfig):
    self.config = config
    self.device = config.device

    if not os.path.exists(self.config.directory):
      raise Exception('raw_dataset.config.directory doesn\'t exist: ' + self.config.directory)

    with utils.Timer('Initializing RawDataset from ' + config.directory):
      with utils.Timer('Loading images from ' + config.directory):
        self.paths = self.load_paths(config)
      print('Found ' + str(len(self.paths)) + ' images', flush=True)
      print('', flush=True)

      self.bucket_resolutions = bucketing_utils.get_closest_bucket_resolutions(config.resolution)

      with utils.Timer('Loading captions & caching image width/height'):
        self.data_items = self.load_data_items(self.config, self.paths, config.directory, self.bucket_resolutions)
      print('', flush=True)

      with utils.Timer('Bucketizing images'):
        self.data_items = self.bucketize_data_items(self.data_items, config.batch_size)

      self._length = len(self.data_items)


  def __len__(self):
    return self._length


  def __getitem__(self, index):
    data_item = self.data_items[index]
    return data_item


  # Load an array of paths to all images in the config.directory
  # If paths_cache.txt exists, loads paths from it
  # Else uses glob and writes paths_cache.txt
  # To get fresh images (skipping cache), set config.ignore_cache to True
  # Return List[str]
  @staticmethod
  def load_paths(config: RawDatasetConfig) -> List[str]:
    if isclose(config.val_split, 0.0, abs_tol=0.001) and config.type == 'val':
      return []

    paths = None
    rnd = random.Random(config.seed)

    paths_cache_path = os.path.join(config.directory, 'paths_cache.txt')
    # paths_cache.txt exists, and we can use cache; make sure the seed is the same; then load the paths and return them
    if not config.ignore_cache and os.path.isfile(paths_cache_path):
      with open(paths_cache_path, 'r', encoding='utf-8') as file:
        seed = file.readline().strip()
        if seed == config.seed:
          paths = [line.strip() for line in file]

    if paths is None:
      # Use glob to load paths; shuffle them with seed; write paths_cache.txt; return paths
      glob_pattern = os.path.join(config.directory, 'data', '*.*')
      paths = glob.glob(glob_pattern)
      paths = [p for p in paths if p.endswith(('jpg', 'jpeg', 'png', 'webp'))]
      random.Random(config.seed).shuffle(paths)
      with open(paths_cache_path, 'w+', encoding='utf-8') as file:
        file.write(str(config.seed) + '\n')
        for path in paths:
          file.write(path.replace('\\', '/') + '\n')

    if isclose(config.val_split, 0.0, abs_tol=0.001):
      return paths

    result = []
    if config.type == 'train':
      for path in paths:
        if random.random() > config.val_split:
          result.append(path)
    else:
      for path in paths:
        if random.random() < config.val_split:
          result.append(path)
    return result


  # Load image width, height, caption, etc. for each path in self.paths; Return List[RawDataItem]
  @staticmethod
  def load_data_items(config: RawDatasetConfig, paths: List[str], directory: str, bucket_resolutions: List[List[int]]) -> List[RawDataItem]:
    meta_directory = os.path.join(directory, 'meta')
    os.makedirs(meta_directory, exist_ok=True)

    result = []

    for path in paths:
      filename_with_ext = path.rsplit('/', 1)[-1]
      filename = filename_with_ext.rsplit('.', 1)[0]
      item_meta_cache_path = os.path.join(meta_directory, filename + '.txt')

      # Get caption
      caption_txt_file_path = os.path.join(directory, 'data', filename + '.txt')
      caption = open(caption_txt_file_path, 'r', encoding='utf-8').read()

      # Load native_width & native_height from cache
      # Still calc width, height to allow for easily changing training resolution
      if os.path.isfile(item_meta_cache_path):
        meta = json.loads(open(item_meta_cache_path, 'r', encoding='utf-8').read())
        width, height = bucketing_utils.find_closest_resolution(bucket_resolutions, meta['native_width'], meta['native_height'])
        data_item = RawDataItem(
          shuffle_captions=config.shuffle_captions,
          cond_dropout=config.cond_dropout,
          path=path,
          caption=caption,
          width=width, height=height,
          native_width=meta['native_width'], native_height=meta['native_height'],
        )
      # Open image; fetch width/height; get caption; cache result
      else:
        # Get image width/height
        image = Image.open(path)
        native_width, native_height = image.size
        width, height = bucketing_utils.find_closest_resolution(bucket_resolutions, native_width, native_height)

        # Save meta cache
        with open(item_meta_cache_path, 'w+', encoding='utf-8') as file:
          file.write(json.dumps({
            'native_width': native_width,
            'native_height': native_height,
          }))

        data_item = RawDataItem(
          shuffle_captions=config.shuffle_captions,
          cond_dropout=config.cond_dropout,
          path=path,
          caption=caption,
          width=width, height=height,
          native_width=native_width, native_height=native_height,
        )

      result.append(data_item)

    return result


  # Sort data items by their target resolution; ensure batch_size consecitive items have the same resolution; return List[RawDataItem]
  @staticmethod
  def bucketize_data_items(data_items: List[RawDataItem], batch_size: int) -> List[RawDataItem]:
    # Sort all data_items by their resolution
    buckets: Dict[Tuple[int, int], List[RawDataItem]] = {}
    for item in data_items:
      if item.size not in buckets:
        buckets[item.size] = []
      buckets[item.size].append(item)

    result_data_items: List[RawDataItem] = []
    buckets_count = 0
    for resolution in buckets:
      bucket_data_items = buckets[resolution]
      bucket_data_items_len = len(bucket_data_items)

      # If > 0, batch_size doesn't fit into the bucket without a remainder,
      # so we need to remove a certain number of items to get an intener number of batches
      items_to_remove_count = bucket_data_items_len % batch_size

      # Not enough images for a batch, skip
      if bucket_data_items_len < batch_size:
        print(f'Bucket {resolution} | {bucket_data_items_len} images total | {bucket_data_items_len} dropped images | Not enough images for a batch, skipping...')
        continue

      if items_to_remove_count > 0:
        bucket_data_items = bucket_data_items[:bucket_data_items_len - items_to_remove_count]
        print(f'Bucket {resolution} | {bucket_data_items_len} images total | {items_to_remove_count} dropped images | {int(len(bucket_data_items) / batch_size)} steps')
      else:
        print(f'Bucket {resolution} | {bucket_data_items_len} images total | No dropped images | {int(len(bucket_data_items) / batch_size)} steps')

      buckets_count += 1
      result_data_items.extend(bucket_data_items)

    print(f'Number of buckets: {buckets_count}')
    print(f'Number of data_items: {len(result_data_items)}')
    print(f'Number of steps: {int(len(result_data_items) / batch_size)}')

    return result_data_items
