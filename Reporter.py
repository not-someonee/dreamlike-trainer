import os

from tqdm import tqdm
import time
from typing import List, Any
from dataclasses import dataclass
import dataclasses
from PIL import Image
import numpy as np
import json5

from accelerate import Accelerator


class AverageMeter:
  def __init__(self, distance=10):
    self.values = []
    self.average = 0
    self.distance = distance

  def reset(self):
    self.values = []
    self.average = 0

  def update(self, value):
    self.values.append(value)
    if len(self.values) > self.distance:
      self.values = self.values[1:]
    self.average = sum(self.values) / len(self.values)


@dataclass
class ReporterConfig:
  accelerator: Accelerator = None
  epochs: int = None
  steps_per_epoch: int = None
  total_steps: int = None
  batch_size: int = None
  run_dir: str = None


class Reporter:
  epochs_progress = None
  total_steps_progress = None
  epoch_progress = None


  def __init__(self, config: ReporterConfig):
    self.config = config
    self.epoch_megapixels_average_meter = AverageMeter()
    self.total_megapixels_average_meter = AverageMeter()
    self.global_step = 0
    self.step = 0
    self.epoch = 0
    self.last_val_loss = 0


  def train_start(self):
    pass


  def train_end(self):
    pass


  def epoch_start(self, epoch: int):
    pass


  def epoch_end(self, epoch: int):
    self.epochs_progress.update(1)
    self.epoch_progress.reset()
    self.total_megapixels_average_meter.reset()


  def step_start(self, epoch: int, step: int):
    if self.epoch_progress is not None:
      self.epoch_progress.desc = 'Epoch ' + str(epoch) + ' steps'


  def step_end(self, epoch: int, step: int, global_step: int, unet_lr: float, te_lr: float, batch, loss: float):
    just_created = False
    if self.epochs_progress is None:
      just_created = True
      self.epochs_progress = tqdm(total=int(self.config.epochs), desc='Epochs', unit='ep')
      self.total_steps_progress = tqdm(total=int(self.config.total_steps), desc='Total steps', unit='it')
      self.epoch_progress = tqdm(total=int(self.config.steps_per_epoch), desc='Epoch steps', unit='it')

    self.global_step = global_step
    self.step = step
    self.epoch = epoch
    self.epoch_megapixels_average_meter.update(batch['megapixels'].item())
    self.total_megapixels_average_meter.update(batch['megapixels'].item())
    self.epochs_progress.update(0)
    self.total_steps_progress.update(1)
    self.epoch_progress.update(1)

    if just_created:
      return

    batch_size = self.config.batch_size

    total_steps_per_sec = float(self.total_steps_progress.format_dict['rate'])
    total_imgs_done = global_step * batch_size
    total_imgs = self.config.total_steps * batch_size
    total_imgs_per_sec = total_steps_per_sec * batch_size
    total_megapx_per_sec = self.total_megapixels_average_meter.average

    steps_per_sec = float(self.epoch_progress.format_dict['rate'])
    imgs_done = step * batch_size
    imgs = self.config.steps_per_epoch * batch_size
    imgs_per_sec = steps_per_sec * batch_size
    megapx_per_sec = self.epoch_megapixels_average_meter.average

    total_postfix = {
      'img done': f'{total_imgs_done}/{total_imgs}',
      'img/s': total_imgs_per_sec,
      'megapx/s': total_megapx_per_sec,
    }
    self.total_steps_progress.set_postfix(**total_postfix)

    self.config.accelerator.log({
      'perf/imgs_per_sec': imgs_per_sec,
      'perf/megapx_per_sec': megapx_per_sec,
      'perf/steps_per_sec': steps_per_sec,
      'hyperparameter/lr_unet': unet_lr,
      'hyperparameter/lr_te': te_lr,
      'epoch_progress': float(step) / float(self.config.steps_per_epoch),
      'loss/train': loss,
    }, step=global_step)

    epoch_postfix = {
      'img done': f'{imgs_done}/{imgs}',
      'img/s': imgs_per_sec,
      'megapx/s': megapx_per_sec,
      'loss': loss,
      'unet lr': unet_lr,
      'te lr': te_lr,
    }
    self.epoch_progress.set_postfix(**epoch_postfix)


  def report_images(self, images: List[Any], gen_id: int, params: List[Any]):
    gens_dir = os.path.join(self.config.run_dir, 'gens')
    os.makedirs(gens_dir, exist_ok=True)

    gen_dir = os.path.join(gens_dir, str(gen_id))
    os.makedirs(gen_dir, exist_ok=True)

    # images_np = [np.array(img) for img in images]
    # images_np = np.stack(images_np, axis=0)
    # images_np = images_np.transpose(0, 3, 1, 2)
    # self.config.accelerator.get_tracker('tensorboard').add_images('images', images_np, global_step=self.global_step)

    for i, image in enumerate(images):
      save_path = os.path.join(gen_dir, str(i) + '.jpg')
      image.save(save_path, 'JPEG', quality=98)

    params_path = os.path.join(gen_dir, 'params.txt')
    params_str = json5.dumps([dataclasses.asdict(p) for p in params], indent=2)
    with open(params_path, 'w', encoding='utf-8') as file:
      file.write(params_str)


  def report_val_loss(self, loss):
    self.config.accelerator.log({ 'loss/val': loss }, step=self.global_step)
    if self.last_val_loss != 0:
      self.config.accelerator.log({ 'loss/val_diff': loss - self.last_val_loss }, step=self.global_step)
    self.last_val_loss = loss
