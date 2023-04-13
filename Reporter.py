import os

from tqdm import tqdm
import time
from dataclasses import dataclass
from PIL import Image

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


class Reporter:
  def __init__(self, config: ReporterConfig):
    self.config = config
    self.epoch_progress = None
    self.total_steps_progress = None
    self.epochs_progress = None
    self.total_steps = None
    self.steps_per_epoch = None
    self.trainer = None
    self.epoch_megapixels_average_meter = AverageMeter()
    self.total_megapixels_average_meter = AverageMeter()
    self.image_num = 0
    self.last_gen_id = -1

  def init(self, trainer, steps_per_epoch, total_steps):
    self.trainer = trainer
    self.steps_per_epoch = steps_per_epoch
    self.total_steps = total_steps
    self.epochs_progress = tqdm(total=int(trainer.config.epochs), desc='Epochs', unit='ep')
    self.total_steps_progress = tqdm(total=int(total_steps), desc='Total steps', unit='it')
    self.epoch_progress = tqdm(total=int(steps_per_epoch), desc='Epoch steps', unit='it')

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

  def step_end(self, epoch: int, step: int, batch, loss: float):
    self.epoch_megapixels_average_meter.update(batch['megapixels'].item())
    self.total_megapixels_average_meter.update(batch['megapixels'].item())
    self.epochs_progress.update(0)
    self.total_steps_progress.update(1)
    self.epoch_progress.update(1)

    batch_size = self.trainer.config.batch_size
    global_step = epoch * self.steps_per_epoch + step

    total_postfix = {
      'img done': f'{(epoch * batch_size + step) * batch_size}/{self.total_steps * batch_size}',
      'img/s': float(self.total_steps_progress.format_dict['rate']) * batch_size,
      'megapx/s': self.total_megapixels_average_meter.average,
    }
    self.total_steps_progress.set_postfix(**total_postfix)

    img_per_sec = float(self.epoch_progress.format_dict['rate']) * batch_size
    megapx_per_sec = self.epoch_megapixels_average_meter.average
    lr = self.trainer.lr_scheduler.get_last_lr()[0]
    steps_per_sec = float(self.epoch_progress.format_dict['rate'])

    self.config.accelerator.log({
      'img_per_sec': img_per_sec,
      'megapx_per_sec': megapx_per_sec,
      'steps_per_sec': steps_per_sec,
      'lr': lr,
      'loss': loss,
    }, step=global_step)

    epoch_postfix = {
      'img done': f'{step * batch_size}/{self.steps_per_epoch * batch_size}',
      'img/s': img_per_sec,
      'megapx/s': megapx_per_sec,
      'loss': loss,
      'lr': lr,
    }
    self.epoch_progress.set_postfix(**epoch_postfix)

  def report_image(self, image: Image, gen_id: int, params: object):
    self.image_num += 1

    if self.last_gen_id != gen_id:
      self.image_num = 0
      self.last_gen_id = gen_id

    gens_dir = os.path.join(self.trainer.config.run_dir, 'gens')
    os.makedirs(gens_dir, exist_ok=True)

    gen_dir = os.path.join(gens_dir, str(gen_id))
    os.makedirs(gen_dir, exist_ok=True)

    save_path = os.path.join(gen_dir, str(self.image_num) + '.jpg')

    image.save(save_path, 'JPEG', quality=98)

    print('', flush=True)
    print(f'Generated image ({params["width"]}x{params["height"]}px)", saved to {save_path}', flush=True)
    print('- Prompt: ' + params['prompt'], flush=True)
    print('- Negative prompt: ' + params['negative_prompt'], flush=True)
    print(f'- Steps/Scale/Seed: {params["steps"]}/{params["scale"]}/{params["seed"]}', flush=True)
    print('', flush=True)


  def report_val(self):
    pass
