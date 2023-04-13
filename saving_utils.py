import torch

import kohya_ss_model_utils
import os.path

from transformers import CLIPTextModel, CLIPModel, CLIPTokenizer, CLIPImageProcessor
from diffusers import AutoencoderKL, UNet2DConditionModel, StableDiffusionPipeline, DDIMScheduler

def save_sd(
  save_path: str,
  tokenizer: CLIPTokenizer,
  text_encoder: CLIPTextModel,
  vae: AutoencoderKL,
  unet: UNet2DConditionModel,
  scheduler: DDIMScheduler,
  should_save_diffusers: bool = True,
  should_save_compvis: bool = False,
  use_safetensors_for_diffusers: bool = True,
  use_safetensors_for_compvis: bool = True,
):
  print('Saving to ' + save_path + '...', flush=True)
  pipe = StableDiffusionPipeline(
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    vae=vae,
    unet=unet,
    scheduler=scheduler,
    requires_safety_checker=False,
    safety_checker=None,
    feature_extractor=None,
  )

  if should_save_diffusers:
    pipe.save_pretrained(save_path, safe_serialization=use_safetensors_for_diffusers)
    print('Saved diffusers to ' + save_path, flush=True)
  if should_save_compvis:
    ext = '.safetensors' if use_safetensors_for_compvis else '.ckpt'
    compvis_save_path = os.path.join(save_path, os.path.basename(save_path) + ext)
    kohya_ss_model_utils.save_stable_diffusion_checkpoint(
      v2=scheduler.config.prediction_type != 'epsilon',
      output_file=compvis_save_path,
      text_encoder=text_encoder,
      vae=vae,
      unet=unet,
      epochs=0,
      steps=0,
      save_dtype=torch.float32,
      ckpt_path=None,
    )
    print('Saved compvis to ' + compvis_save_path, flush=True)

