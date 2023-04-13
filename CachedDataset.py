from typing import List
import os
from dataclasses import dataclass
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL
import safetensors

from RawDataset import RawDataset, RawDataItem
import utils
import sd_utils


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
    self.getitem_call_num = 0


  @staticmethod
  def make_collate_fn(tokenizer: CLIPTokenizer, device):
    @torch.no_grad()
    def collate_fn(batch):
      latents = [v['image_latent'] for v in batch]
      caption_token_ids = [sd_utils.tokenize(tokenizer, v['caption'], device) for v in batch]
      megapixels = sum([v['megapixels'] for v in batch])

      del batch

      latents = torch.stack(latents).float()
      caption_token_ids = torch.stack(caption_token_ids)

      return { 'latents': latents, 'caption_token_ids': caption_token_ids, 'megapixels': megapixels }

    return collate_fn


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

    self.getitem_call_num += 1
    if self.getitem_call_num >= self.__len__() and self.raw_dataset.config.ignore_cache:
      self.raw_dataset.config.ignore_cache = False
      print('Recalculated cache, switching ignore_cache to False')

    return cache

