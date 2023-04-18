from typing import Literal
import time
import os

import torch

from Reporter import Reporter
from RawDataset import RawDataset
import saving_utils

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel, AutoencoderKL, EulerDiscreteScheduler
from accelerate import Accelerator

from dataclasses import dataclass


@dataclass
class SaverConfig:
  save_every_n_minutes: float = 60.0
  save_every_n_epochs: int = 1
  use_safetensors: bool = True
  save_compvis_checkpoint: bool = False
  use_safetensors_for_compvis: bool = True

  pretrained_model_name_or_path: str = None
  save_dir: str = None
  checkpoint_name: str = None
  vae: AutoencoderKL = None
  unet: UNet2DConditionModel = None
  text_encoder: CLIPTextModel = None
  tokenizer: CLIPTokenizer = None
  scheduler: EulerDiscreteScheduler = None
  accelerator: Accelerator = None


class Saver:
  epoch: int = 0
  step: int = 0
  global_step: int = 0

  def __init__(self, config: SaverConfig):
    self.config = config
    self.last_saved_at = time.time()


  def train_start(self):
    pass


  def train_end(self):
    self.save()


  def epoch_start(self, epoch: int):
    pass


  def epoch_end(self, epoch: int):
    if epoch == 0:
      if self.config.save_every_n_epochs == 1:
        self.save()
    elif (epoch % self.config.save_every_n_epochs) == 0:
      self.save()


  def step_start(self, epoch: int, step: int):
    pass


  def step_end(self, epoch: int, step: int, global_step: int, unet_lr: float, te_lr: float, batch, loss: float):
    self.epoch = epoch
    self.step = step
    self.global_step = global_step
    if (time.time() - self.last_saved_at) > (self.config.save_every_n_minutes * 60.0):
      self.save()


  def get_checkpoint_name(self):
    return self.config.checkpoint_name + '__ep_' + str(self.epoch) + '__s' + str(self.global_step)


  def save(self):
    self.config.accelerator.wait_for_everyone()
    if self.config.accelerator.is_main_process:
      self.last_saved_at = time.time()
      save_path = os.path.join(self.config.save_dir, self.get_checkpoint_name())
      saving_utils.save_sd(
        save_path=save_path,
        tokenizer=self.config.tokenizer,
        text_encoder=self.config.accelerator.unwrap_model(self.config.text_encoder),
        vae=AutoencoderKL.from_pretrained(self.config.pretrained_model_name_or_path, subfolder='vae').to(self.config.accelerator.device),
        unet=self.config.accelerator.unwrap_model(self.config.unet),
        scheduler=self.config.scheduler,
        should_save_diffusers=True,
        should_save_compvis=self.config.save_compvis_checkpoint,
        use_safetensors_for_diffusers=self.config.use_safetensors,
        use_safetensors_for_compvis=self.config.use_safetensors_for_compvis,
      )