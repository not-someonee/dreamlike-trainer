import tqdm

import utils
import train_utils
from Reporter import Reporter

from dataclasses import dataclass
import time

import torch
from torch.utils.data import DataLoader
from pytorch_lightning.utilities.seed import isolate_rng


@dataclass
class ValidatorConfig:
  calc_loss_fn: None
  dataloader: DataLoader = None
  reporter: Reporter = None
  seed: int = None

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
    if global_step != 0 and (global_step % self.config.validate_every_n_steps) == 0:
      self.val()


  @torch.no_grad()
  def val(self):
    print('\n\n\n', flush=True)
    with isolate_rng(), utils.Timer('Validation'):
      print('\n\n', flush=True)
      torch.manual_seed(self.config.seed + 34573)

      losses = []
      with tqdm.tqdm(total=len(self.config.dataloader), desc='Validation', unit='it') as pbar:
        for step, batch in enumerate(self.config.dataloader):
          loss = self.config.calc_loss_fn(step, batch, 'val').item()
          losses.append(loss)
          pbar.update(1)
        pbar.close()

      average_loss = sum(losses) / len(losses)

      self.config.reporter.report_val_loss(average_loss)

    self.last_val_at = time.time()
    print('\n\n', flush=True)
