import os
import random
import saving_utils
import utils

import torch
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, EulerDiscreteScheduler


cache_file_path = './models_cache.txt'


def load_sd(pretrained_model_name_or_path: str, device, use_cache: bool = True):
  if not use_cache:
    return _load_sd(pretrained_model_name_or_path, device)

  with open(cache_file_path, 'r') as file:
    cache = file.read().splitlines()

  for line in cache:
    cached_model_name, cached_model_directory = line.split(':')
    if pretrained_model_name_or_path == cached_model_name:
      return _load_sd(os.path.join('./models_cache', cached_model_directory), device)

  sd = _load_sd(pretrained_model_name_or_path, device)

  basename = os.path.basename(pretrained_model_name_or_path) + '_' + str(random.randint(0, 99999999))
  saving_utils.save_sd(
    save_path=os.path.join('./models_cache', basename),
    tokenizer=sd[0],
    text_encoder=sd[1],
    unet=sd[2],
    vae=sd[3],
    scheduler=DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder='scheduler'),
    should_save_diffusers=True,
    should_save_compvis=False,
    use_safetensors_for_diffusers=True,
    use_safetensors_for_compvis=True,
  )
  with open(cache_file_path, 'a') as cache_file:
    cache_file.write(f'{pretrained_model_name_or_path}:{basename}\n')

  return sd


def _load_sd(pretrained_model_name_or_path: str, device):
  with utils.Timer('Loading SD'):
    with utils.Timer('Loading tokenizer'):
      tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder='tokenizer')
      tokenizer.deprecation_warnings['sequence-length-is-longer-than-the-specified-maximum'] = True
    with utils.Timer('Loading text_encoder'):
      text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder='text_encoder', torch_dtype=torch.float32)
      text_encoder.to(device)
    with utils.Timer('Loading unet'):
      unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder='unet', torch_dtype=torch.float32)
      unet.enable_xformers_memory_efficient_attention()
      unet.to(device)
    with utils.Timer('Loading vae'):
      vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.float16, subfolder='vae')
      vae.enable_xformers_memory_efficient_attention()
      vae.to(device)
      vae.requires_grad_(False)
    scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder='scheduler')

  return tokenizer, text_encoder, unet, vae, scheduler


def tokenize(tokenizer: CLIPTokenizer, caption: str, device) -> torch.Tensor:
  return torch.tensor(tokenizer(
    caption,
    padding='max_length',
    truncation=True,
    max_length=tokenizer.model_max_length,
  ).input_ids, device=device)


# Decode latents to pytorch tensor
def decode_latents(vae: AutoencoderKL, latents: torch.Tensor) -> torch.Tensor:
  latents = 1 / 0.18215 * latents
  image = vae.decode(latents).sample
  image = (image / 2 + 0.5).clamp(0, 1)
  image = image.cpu().permute(0, 2, 3, 1).float()
  return image

