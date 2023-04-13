import os

from tqdm import tqdm
import time
from dataclasses import dataclass
from PIL import Image
import numpy as np

from accelerate import Accelerator


class AverageMeter:
  def __init__(self, distance=10):
    self.sum = 0
    self.values = []
    self.average = 0
    self.distance = distance
    self.started_at = time.time()

  def reset(self):
    self.sum = 0
    self.average = 0
    self.started_at = time.time()

  def update(self, value):
    self.sum += value
    now = time.time()
    diff = now - self.started_at
    self.average = self.sum / diff


# class AverageMeter:
#   def __init__(self):
#     self.sum = 0
#     self.average = 0
#     self.started_at = time.time()
#
#   def reset(self):
#     self.sum = 0
#     self.average = 0
#     self.started_at = time.time()
#
#   def update(self, value):
#     self.sum += value
#     now = time.time()
#     diff = now - self.started_at
#     self.average = self.sum / diff


@dataclass
class ReporterConfig:
  accelerator: Accelerator = None
  lr_scheduler = None
  epochs: int = None
  steps_per_epoch: int = None
  total_steps: int = None
  batch_size: int = None
  run_dir: str = None


class Reporter:
  def __init__(self, config: ReporterConfig):
    self.config = config
    self.epoch_megapixels_average_meter = AverageMeter()
    self.total_megapixels_average_meter = AverageMeter()
    self.epochs_progress = tqdm(total=int(config.epochs), desc='Epochs', unit='ep')
    self.total_steps_progress = tqdm(total=int(config.total_steps), desc='Total steps', unit='it')
    self.epoch_progress = tqdm(total=int(config.steps_per_epoch), desc='Epoch steps', unit='it')
    self.image_num = 0
    self.last_gen_id = -1
    self.images = []
    self.global_step = 0
    self.step = 0
    self.epoch = 0


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
    self.epoch_progress.desc = 'Epoch ' + str(epoch) + ' steps'


  def step_end(self, epoch: int, step: int, global_step: int, lr: float, batch, loss: float):
    self.global_step = global_step
    self.step = step
    self.epoch = epoch
    self.epoch_megapixels_average_meter.update(batch['megapixels'].item())
    self.total_megapixels_average_meter.update(batch['megapixels'].item())
    self.epochs_progress.update(0)
    self.total_steps_progress.update(1)
    self.epoch_progress.update(1)

    batch_size = self.config.batch_size

    total_steps_per_sec = float(self.total_steps_progress.format_dict['rate'])
    total_imgs_done = global_step * batch_size
    total_imgs = self.config.total_steps * batch_size
    total_imgs_per_sec = total_steps_per_sec * batch_size
    total_megapx_per_sec = self.total_megapixels_average_meter.average

    steps_per_sec = float(self.epoch_progress.format_dict['rate'])
    imgs_done = step * batch_size
    imgs = self.steps_per_epoch * batch_size
    imgs_per_sec = steps_per_sec * batch_size
    megapx_per_sec = self.epoch_megapixels_average_meter.average

    total_postfix = {
      'img done': f'{total_imgs_done}/{total_imgs}',
      'img/s': total_imgs_per_sec,
      'megapx/s': total_megapx_per_sec,
    }
    self.total_steps_progress.set_postfix(**total_postfix)

    self.config.accelerator.log({
      'imgs_per_sec': imgs_per_sec,
      'megapx_per_sec': megapx_per_sec,
      'steps_per_sec': steps_per_sec,
      'lr': lr,
      'loss': loss,
    }, step=global_step)

    epoch_postfix = {
      'img done': f'{imgs_done}/{imgs}',
      'img/s': imgs_per_sec,
      'megapx/s': megapx_per_sec,
      'loss': loss,
      'lr': lr,
    }
    self.epoch_progress.set_postfix(**epoch_postfix)


  def report_image(self, image: Image, gen_id: int, params: object):
    self.image_num += 1

    if self.last_gen_id != gen_id:
      images = [np.array(img) for img in self.images]
      images_np = np.stack(images, axis=0)
      images_np = images_np.transpose(0, 3, 1, 2)
      self.config.accelerator.get_tracker('tensorboard').add_images('images', images_np, global_step=self.global_step)
      self.image_num = 0
      self.last_gen_id = gen_id
      self.images = []

    gens_dir = os.path.join(self.config.run_dir, 'gens')
    os.makedirs(gens_dir, exist_ok=True)

    gen_dir = os.path.join(gens_dir, str(gen_id))
    os.makedirs(gen_dir, exist_ok=True)

    save_path = os.path.join(gen_dir, str(self.image_num) + '.jpg')

    image.save(save_path, 'JPEG', quality=98)
    self.images.append(image)

    print('', flush=True)
    print(f'Generated image ({params["width"]}x{params["height"]}px)", saved to {save_path}', flush=True)
    print('- Prompt: ' + params['prompt'], flush=True)
    print('- Negative prompt: ' + params['negative_prompt'], flush=True)
    print(f'- Steps/Scale/Seed: {params["steps"]}/{params["scale"]}/{params["seed"]}', flush=True)
    print('', flush=True)


  def report_val(self):
    pass
