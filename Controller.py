import time
from keyboard_util import get_pressed_keys
from bcolors import bcolors
from dataclasses import dataclass

from Reporter import Reporter
from RawDataset import RawDataset

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel, AutoencoderKL
from accelerate import Accelerator



class Controller:
  def __init__(self, trainer):
    self.trainer = trainer
    self.print_help_every_n_minutes = 15
    self.last_printed_help_at = time.time()
    self.save_at_the_end_of_epoch = False
    self.gen_at_the_end_of_epoch = False
    self.stop_at_the_end_of_epoch = False

  def train_start(self):
    self.print_help()

  def train_end(self):
    pass

  def epoch_start(self, epoch: int):
    pass

  def epoch_end(self, epoch: int):
    if self.gen_at_the_end_of_epoch:
      self.gen(at_the_end_of_epoch=False)

    if self.stop_at_the_end_of_epoch:
      self.stop(at_the_end_of_epoch=False)
    elif self.save_at_the_end_of_epoch:
      self.save(at_the_end_of_epoch=False)

  def step_start(self, epoch: int, step: int):
    pass

  def step_end(self, epoch: int, step: int, global_step: int, unet_lr: float, te_lr: float, batch, loss: float):
    if (time.time() - self.last_printed_help_at) > (self.print_help_every_n_minutes * 60):
      self.print_help()
    self.check_keypresses()


  def check_keypresses(self):
    keys = get_pressed_keys()

    if 'w' in keys:
      self.save(at_the_end_of_epoch=False)
    elif 'e' in keys:
      self.gen(at_the_end_of_epoch=False)
    elif 'p' in keys:
      self.stop(at_the_end_of_epoch=False)
    elif 's' in keys:
      self.save(at_the_end_of_epoch=True)
    elif 'd' in keys:
      self.gen(at_the_end_of_epoch=True)
    elif 'k' in keys:
      self.stop(at_the_end_of_epoch=True)
    elif 'h' in keys:
      self.print_help()


  def save(self, at_the_end_of_epoch):
    if at_the_end_of_epoch:
      msg = 'Will save a checkpoint at the end of this epoch.' if not self.save_at_the_end_of_epoch else 'Cancelled saving a checkpoint at the end of this epoch.'
      print('\n' + msg + '\n', flush=True)
      self.save_at_the_end_of_epoch = not self.save_at_the_end_of_epoch
      return

    self.trainer.saver.save()
    self.save_at_the_end_of_epoch = False


  def gen(self, at_the_end_of_epoch):
    if at_the_end_of_epoch:
      msg = 'Will generate images at the end of this epoch.' if not self.save_at_the_end_of_epoch else 'Cancelled generating images at the end of this epoch.'
      print('\n' + msg + '\n', flush=True)
      self.gen_at_the_end_of_epoch = not self.gen_at_the_end_of_epoch
      return

    self.trainer.imagen.gen()
    self.gen_at_the_end_of_epoch = False


  def stop(self, at_the_end_of_epoch):
    if at_the_end_of_epoch:
      msg = 'Will save a checkpoint and stop training at the end of this epoch.' if not self.save_at_the_end_of_epoch else 'Cancelled saving a checkpoint and stopping training at the end of this epoch.'
      print('\n' + msg + '\n', flush=True)
      self.stop_at_the_end_of_epoch = not self.stop_at_the_end_of_epoch
      return

    self.trainer.stop()


  def print_help(self):
    self.last_printed_help_at = time.time()
    print('\n\n', flush=True)
    print('\n\n', flush=True)
    print(f'{bcolors.WARNING}Press W to save a checkpoint now{bcolors.ENDC}', flush=True)
    print(f'{bcolors.WARNING}Press E to generate images now{bcolors.ENDC}', flush=True)
    print(f'{bcolors.WARNING}Press P to save a checkpoint and stop training now{bcolors.ENDC}', flush=True)
    print('', flush=True)
    print(f'{bcolors.WARNING}Press A to save a checkpoint at the end of this epoch{bcolors.ENDC}', flush=True)
    print(f'{bcolors.WARNING}Press S to generate images at the end of this epoch{bcolors.ENDC}', flush=True)
    print(f'{bcolors.WARNING}Press K to save a checkpoint and stop training at the end of this epoch{bcolors.ENDC}', flush=True)
    print('', flush=True)
    print(f'{bcolors.WARNING}Press H to print this message{bcolors.ENDC}', flush=True)
    print('\n\n', flush=True)
