import tqdm

import utils
import train_utils
from Reporter import Reporter

from dataclasses import dataclass
import time

import torch
from torch.utils.data import DataLoader
from pytorch_lightning.utilities.seed import isolate_rng
from accelerate import Accelerator
from diffusers import UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel


@dataclass
class ValidatorConfig:
  calc_loss_fn: None
  dataloader: DataLoader = None
  reporter: Reporter = None
  seed: int = None

  accelerator: Accelerator = None
  unet: UNet2DConditionModel = None
  text_encoder: CLIPTextModel = None
  vae: AutoencoderKL = None
  validate_every_n_minutes: int = 30
  validate_every_n_epochs: int = 99999999
  validate_every_n_steps: int = 99999999
  validate_at_training_start: bool = True
  validate_at_training_end: bool = True


class Validator:
  def __init__(self, config: ValidatorConfig):
    self.config = config
    self.last_val_at = time.time()

  def train_start(self):
    if self.config.validate_at_training_start:
      self.val()


  def train_end(self):
    if self.config.validate_at_training_end:
      self.val()


  def epoch_start(self, epoch: int):
    pass


  def epoch_end(self, epoch: int):
    if epoch == 0:
      if self.config.validate_every_n_epochs == 1:
        self.val()
    elif (epoch % self.config.validate_every_n_epochs) == 0:
      self.val()


  def step_start(self, epoch: int, step: int):
    pass


  def step_end(self, epoch: int, step: int, global_step: int, unet_lr: float, te_lr: float, batch, loss: float):
    if (time.time() - self.last_val_at) > (self.config.validate_every_n_minutes * 60.0):
      self.val()
    elif global_step != 0 and (global_step % self.config.validate_every_n_steps) == 0:
      self.val()

  #TODO: Duplicates function in DreamlikeTrainer.
  @torch.no_grad()
  def prepare_batch(self, batch):
    latents = []
    for pixel_values in batch['latents']:
      image_latent = self.config.vae.encode(
        pixel_values.unsqueeze(0).to(self.config.vae.device, dtype=torch.float16)).latent_dist.sample() * 0.18215
      image_latent = image_latent.squeeze(0).to('cpu')
      latents.append(image_latent)
      del pixel_values

    batch['latents'] = torch.stack(latents).float()
    del latents

  @torch.no_grad()
  def val(self):
    utils.garbage_collect()
    if self.config.accelerator.is_main_process:
      print('\n\n\n', flush=True)
      with isolate_rng(), utils.Timer('Validation'):
        pass
        print('\n\n', flush=True)
        torch.manual_seed(self.config.seed + 34573)

        losses_with_snr = [1]
        losses = [1]
        with tqdm.tqdm(total=len(self.config.dataloader), desc='Validation', unit='it',  disable=not self.config.accelerator.is_main_process) as pbar:
          for step, batch in enumerate(self.config.dataloader):
            self.prepare_batch(batch)
            loss, loss_with_snr = self.config.calc_loss_fn(step, batch, self.config.unet, self.config.text_encoder, 'val')
            # loss, loss_with_snr = self.config.accelerator.gather_for_metrics((loss, loss_with_snr))
            losses.append(loss.mean().item())
            losses_with_snr.append(loss_with_snr.mean().item())
            pbar.update(1)
          pbar.close()

        average_loss = sum(losses) / len(losses)
        average_loss_with_snr = sum(losses_with_snr) / len(losses_with_snr)

        self.config.reporter.report_val_loss(average_loss, average_loss_with_snr)

      self.last_val_at = time.time()
      print('\n\n', flush=True)
    utils.garbage_collect()
