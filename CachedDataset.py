from typing import List
from torch.utils.data import Dataset
import os
from dataclasses import dataclass
import safetensors

import torch
import torch.nn as nn
from transformers import CLIPTextModel
from diffusers import AutoencoderKL

import utils
from RawDataset import RawDataset, RawDataItem
import time


@dataclass
class CachedDatasetConfig:
  raw_dataset: RawDataset
  vae: AutoencoderKL


class CachedDataset(Dataset):
  def __init__(self, config: CachedDatasetConfig):
    self.config = config
    self.raw_dataset = config.raw_dataset
    self.raw_config = config.raw_dataset.config
    self.device = self.raw_config.device

  def __len__(self):
    return self.raw_dataset.__len__()

  @torch.no_grad()
  def __getitem__(self, index):
    cache_dir = os.path.join(self.raw_config.directory, 'cache')
    os.makedirs(cache_dir, exist_ok=True)

    data_item: RawDataItem = self.config.raw_dataset.__getitem__(index)

    filename_with_ext = data_item.path.rsplit('/', 1)[-1]
    filename = filename_with_ext.rsplit('.', 1)[0]
    cache_tensor_path = os.path.join(cache_dir, filename + '.safetensors')

    if os.path.isfile(cache_tensor_path) and not self.raw_config.ignore_cache:
      cache = {}
      with safetensors.safe_open(cache_tensor_path, framework='pt') as f:
        for key in f.keys():
          cache[key] = f.get_tensor(key)
      cache['caption'] = data_item.get_caption()
    else:
      pixel_values, caption = data_item.get_data()
      image_latent = self.config.vae.encode(pixel_values.unsqueeze(0).to(self.device, dtype=torch.float16)).latent_dist.sample() * 0.18215
      del pixel_values
      cache = {
        'image_latent': image_latent.squeeze(0).to('cpu'),
        'megapixels': torch.tensor(data_item.width * data_item.height / 1_000_000),
      }
      del image_latent
      safetensors.torch.save_file(cache, cache_tensor_path)
      cache['caption'] = caption

    return cache

