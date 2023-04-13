import time
import keyboard
from dataclasses import dataclass

from Reporter import Reporter
from RawDataset import RawDataset
from DreamlikeTrainer import DreamlikeTrainer

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel, AutoencoderKL
from accelerate import Accelerator


class Controller:
  def __init__(self, trainer: DreamlikeTrainer):
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

  def step_end(self, epoch: int, step: int, batch, loss: float):
    if (time.time() - self.last_printed_help_at) > (self.print_help_every_n_minutes * 60):
      self.print_help()
    self.check_keypresses()


  def check_keypresses(self):
    if not keyboard.is_pressed('ctrl') or not keyboard.is_pressed('alt'):
      return

    shift = keyboard.is_pressed('shift')
    s = keyboard.is_pressed('s')
    g = keyboard.is_pressed('g')
    q = keyboard.is_pressed('q')

    if s:
      self.save(at_the_end_of_epoch=shift)
    elif g:
      self.gen(at_the_end_of_epoch=shift)
    elif q:
      self.save(at_the_end_of_epoch=shift)
      self.stop(at_the_end_of_epoch=shift)


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


  @staticmethod
  def print_help():
    self.last_printed_help_at = time.time()
    print('', flush=True)
    print('Press CTRL+ALT+S to save a checkpoint now', flush=True)
    print('Press CTRL+ALT+G to generate images now', flush=True)
    print('Press CTRL+ALT+Q to save a checkpoint and stop training now', flush=True)
    print('', flush=True)
    print('Press CTRL+SHIFT+ALT+S to save a checkpoint at the end of this epoch', flush=True)
    print('Press CTRL+SHIFT+ALT+G to generate images at the end of this epoch', flush=True)
    print('Press CTRL+SHIFT+ALT+Q to save a checkpoint and stop training at the end of this epoch', flush=True)
    print('', flush=True)
    print(f'Press CTRL+H to print this message', flush=True)
    print('', flush=True)
