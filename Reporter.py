import os

from tqdm import tqdm
import time
from typing import List, Any
from dataclasses import dataclass
import dataclasses
from PIL import Image
import numpy as np
import json5
import math

from accelerate import Accelerator


def create_grid(images, grid_size):
  width, height = images[0].size

  grid_width = width * grid_size[0]
  grid_height = height * grid_size[1]

  grid_image = Image.new('RGB', (grid_width, grid_height))

  for i in range(grid_size[1]):
    for j in range(grid_size[0]):
      image_idx = i * grid_size[0] + j
      if image_idx < len(images):
        grid_image.paste(images[image_idx], (j * width, i * height))
      else:
        break

  return grid_image


def get_grid_size(num_images, max_columns):
  columns = min(num_images, max_columns)
  rows = math.ceil(num_images / max_columns)
  return columns, rows


class AverageMeter:
  def __init__(self, distance=10):
    self.values = []
    self.dates = []
    self.average = 0
    self.distance = distance

  def reset(self):
    self.values = []
    self.dates = []
    self.average = 0

  def update(self, value):
    self.values.append(value)
    self.dates.append(time.time())
    if len(self.values) > self.distance:
      self.values = self.values[1:]
      self.dates = self.dates[1:]
    self.average = sum(self.values) / (time.time() - self.dates[0])


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
    self.loss_average_meter = AverageMeter()
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
    if self.config.accelerator.is_main_process:
      self.epochs_progress.update(1)
      self.epoch_progress.reset()
      self.total_megapixels_average_meter.reset()


  def step_start(self, epoch: int, step: int):
    if self.epoch_progress is not None:
      self.epoch_progress.desc = 'Epoch ' + str(epoch) + ' steps'


  def step_end(self, epoch: int, step: int, global_step: int, unet_lr: float, te_lr: float, batch, loss: float):
    if not self.config.accelerator.is_main_process:
      return
    
    just_created = False
    if self.epochs_progress is None:
      just_created = True
      self.epochs_progress = tqdm(total=int(self.config.epochs), desc='Epochs', unit='ep')
      self.total_steps_progress = tqdm(total=int(self.config.total_steps), desc='Total steps', unit='it')
      self.epoch_progress = tqdm(total=int(self.config.steps_per_epoch), desc='Epoch steps', unit='it')

    self.global_step = global_step
    self.step = step
    self.epoch = epoch
    self.epoch_megapixels_average_meter.update(batch['megapixels'])
    self.total_megapixels_average_meter.update(batch['megapixels'])
    self.loss_average_meter.update(loss)
    self.epochs_progress.update(0)
    self.total_steps_progress.update(1)
    self.epoch_progress.update(1)

    if just_created:
      return

    actual_batch_size = self.config.batch_size * self.config.accelerator.num_processes

    total_steps_per_sec = float(self.total_steps_progress.format_dict['rate'])
    total_imgs_done = global_step * actual_batch_size
    total_imgs = self.config.total_steps * actual_batch_size
    total_imgs_per_sec = total_steps_per_sec * actual_batch_size
    total_megapx_per_sec = self.total_megapixels_average_meter.average

    steps_per_sec = float(self.epoch_progress.format_dict['rate'])
    imgs_done = step * actual_batch_size
    imgs = self.config.steps_per_epoch * actual_batch_size
    imgs_per_sec = steps_per_sec * actual_batch_size
    megapx_per_sec = self.epoch_megapixels_average_meter.average

    total_postfix = {
      'img done': f'{total_imgs_done}/{total_imgs}',
      'img/s': total_imgs_per_sec,
      'megapx/s': total_megapx_per_sec,
    }
    self.total_steps_progress.set_postfix(**total_postfix)

    if len(self.total_megapixels_average_meter.values) == self.total_megapixels_average_meter.distance:
      self.config.accelerator.log({
        'perf/megapx_per_sec': megapx_per_sec,
        'perf/imgs_per_sec': imgs_per_sec,
        'perf/steps_per_sec': steps_per_sec,
      }, step=global_step)

    self.config.accelerator.log({
      'hyperparameter/lr_unet': unet_lr,
      'hyperparameter/lr_te': te_lr,
      'epoch': epoch,
      'epoch_progress': float(step) / float(self.config.steps_per_epoch),
      'loss/train': loss,
      'loss/train_avg': self.loss_average_meter.average,
    }, step=global_step)

    epoch_postfix = {
      'img done': f'{imgs_done}/{imgs}',
      'img/s': imgs_per_sec,
      'megapx/s': megapx_per_sec,
      'unet lr': unet_lr,
      'te lr': te_lr,
    }
    self.epoch_progress.set_postfix(**epoch_postfix)


  def report_images(self, images: List[Any], prefix: str, gen_id: int, params: List[Any]):
    gens_dir = os.path.join(self.config.run_dir, 'gens')
    os.makedirs(gens_dir, exist_ok=True)

    gen_dir = os.path.join(gens_dir, 'step_' + str(self.global_step))
    os.makedirs(gen_dir, exist_ok=True)

    # images_np = [np.array(img) for img in images]
    # images_np = np.stack(images_np, axis=0)
    # images_np = images_np.transpose(0, 3, 1, 2)
    # self.config.accelerator.get_tracker('tensorboard').add_images('images', images_np, global_step=self.global_step)

    grid = create_grid(images, get_grid_size(len(images), 7))

    grid_save_path = os.path.join(gen_dir, 'grid_' + prefix + os.path.basename(self.config.run_dir) + '.jpg')
    grid.save(grid_save_path, 'JPEG', quality=98)

    for i, image in enumerate(images):
      save_path = os.path.join(gen_dir, prefix + str(i) + '.jpg')
      image.save(save_path, 'JPEG', quality=98)

    params_path = os.path.join(gen_dir, 'params.txt')
    params_str = json5.dumps([dataclasses.asdict(p) for p in params], indent=2)
    with open(params_path, 'w', encoding='utf-8') as file:
      file.write(params_str)


  def report_val_loss(self, loss, loss_with_snr):
    if self.config.accelerator.is_main_process:
      self.config.accelerator.log({
        'loss/val': loss_with_snr,
        'loss/val_no_snr': loss,
      }, step=self.global_step)
      self.last_val_loss = loss
