import setup_env

import utils
from RawDataset import RawDataset, RawDatasetConfig
from CachedDataset import CachedDataset, CachedDatasetConfig
from Reporter import ReporterConfig, Reporter
from Imagen import ImagenConfig, Imagen
from Saver import SaverConfig, Saver
from Validator import ValidatorConfig, Validator
from Controller import Controller
import saving_utils
import train_utils
import sd_utils

from typing import Literal, List
import dataclasses
from dataclasses import dataclass
import gc
import math
import os
import builtins
import tqdm
import itertools
import shutil

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from transformers import CLIPTextModel, CLIPModel, CLIPTokenizer, CLIPImageProcessor
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, EulerDiscreteScheduler
from diffusers.optimization import get_scheduler as get_lr_scheduler
from lion_pytorch import Lion
from torch.optim.lr_scheduler import LambdaLR

from accelerate import Accelerator
from accelerate.utils import set_seed, broadcast_object_list

import bitsandbytes


@dataclass
class DreamlikeTrainerConfig:
  pretrained_model_name_or_path: str
  dataset_dir: str
  mode: Literal['train', 'lr_finder'] = 'train'
  cache_models: bool = True
  epochs: int = 4
  batch_size: int = 3
  gradient_accumulation_steps: int = 2
  seed: int = 42
  clip_penultimate: bool = False
  cond_dropout: float = 0.1
  shuffle_captions: bool = True
  shuffle_dataset_each_epoch: bool = True
  offset_noise_weight: float = 0.07

  dataset_max_images: int = 0

  dataset_val_split: float = 0.1
  dataset_max_val_images: int = 500
  validate_every_n_minutes: int = 30
  validate_every_n_epochs: int = 99999999
  validate_every_n_steps: int = 99999999
  validate_at_training_start: bool = True
  validate_at_training_end: bool = True

  unet_lr: float = 1e-6
  unet_lr_scheduler: Literal['constant', 'cosine', 'cosine_with_restarts'] = 'cosine_with_restarts'
  unet_lr_warmup_steps: int = 0
  unet_lr_epochs: int = 0

  te_lr: float = 1e-6
  te_lr_scheduler: Literal['constant', 'cosine', 'cosine_with_restarts'] = 'cosine_with_restarts'
  te_lr_warmup_steps: int = 0
  te_lr_epochs: int = 0

  optimizer: Literal['adam', 'lion'] = 'lion'

  adam_optimizer_weight_decay: float = 1e-2
  adam_optimizer_beta_one: float = 0.9
  adam_optimizer_beta_two: float = 0.999

  lion_optimizer_weight_decay: float = 1e-2
  lion_optimizer_beta_one: float = 0.9
  lion_optimizer_beta_two: float = 0.99
  lion_optimizer_lr_multiplier: float = 0.2

  use_snr: bool = True
  snr: float = 5.0
  snr_warmup_steps: int = 0
  resolution: int = 768
  precache_latents: bool = False
  ignore_cache: bool = False
  config_path: str = None
  run_name: str = ''


class DreamlikeTrainer:
  config: DreamlikeTrainerConfig = None
  run_name: str = None
  run_dir: str = None

  current_step: int = 0
  global_step: int = 0
  epoch: int = 0
  last_loss: float = None

  steps_per_epoch: int = None
  total_steps: int = None

  tokenizer: CLIPTokenizer = None
  text_encoder: CLIPTextModel = None
  unet: UNet2DConditionModel = None
  vae: AutoencoderKL = None
  scheduler: DDPMScheduler = None
  imagen_scheduler: EulerDiscreteScheduler = None

  accelerator: Accelerator = None
  device = None

  unet_lr_scheduler = None
  te_lr_scheduler = None

  unet_optimizer = None
  te_optimizer = None

  raw_dataset_train: RawDataset = None
  raw_dataset_val: RawDataset = None

  cached_dataset_train: CachedDataset = None
  cached_dataset_val: CachedDataset = None
  cached_dataloader_train: DataLoader = None
  cached_dataloader_val: DataLoader = None

  reporter_config: ReporterConfig = None
  imagen_config: ImagenConfig = None

  reporter: Reporter = None
  imagen: Imagen = None
  saver: Saver = None
  controller: Controller = None
  validator: Validator = None
  modules = List = None

  _stop: bool = False


  def __init__(self, config: DreamlikeTrainerConfig, reporter_config: ReporterConfig, imagen_config: ImagenConfig, saver_config: SaverConfig):
    self.config = config
    self.reporter_config = reporter_config
    self.imagen_config = imagen_config
    self.saver_config = saver_config

    if self.config.optimizer == 'lion':
      self.config.unet_lr *= self.config.lion_optimizer_lr_multiplier
      self.config.te_lr *= self.config.lion_optimizer_lr_multiplier

    self.create_accelerator()
    self.setup_run_dir()

    self.load_sd()

    set_seed(config.seed)

    if not self.accelerator.is_main_process:
      self.accelerator.wait_for_everyone()
    self.load_raw_datasets()
    self.load_cached_datasets()
    self.load_cached_dataloaders()
    if self.accelerator.is_main_process:
      self.accelerator.wait_for_everyone()

    if self.config.precache_latents:
      self.cached_dataset_train.precache()
      self.cached_dataset_val.precache()

    self.create_optimizer()
    self.create_lr_scheduler()

    self.prepare_accelerator()

    self.create_reporter()
    self.create_imagen()
    self.create_saver()
    self.create_controller()
    self.create_validator()

    self.modules = [self.reporter, self.imagen, self.saver, self.controller, self.validator]


  def train_start(self):
    for module in self.modules:
      module.train_start()


  def train_end(self):
    for module in self.modules:
      module.train_end()


  def epoch_start(self, epoch: int):
    for module in self.modules:
      module.epoch_start(epoch)


  def epoch_end(self, epoch: int):
    for module in self.modules:
      module.epoch_end(epoch)


  def step_start(self, epoch: int, step: int):
    self.current_step = step
    self.global_step = epoch * self.steps_per_epoch + step
    for module in self.modules:
      module.step_start(epoch, step)


  def step_end(self, epoch: int, step: int, batch):
    unet_lr = self.unet_lr_scheduler.get_last_lr()[0]
    te_lr = self.te_lr_scheduler.get_last_lr()[0]
    kwargs = { 'epoch': epoch, 'step': step, 'batch': batch, 'loss': self.last_loss, 'unet_lr': unet_lr, 'te_lr': te_lr, 'global_step': self.global_step }
    for module in self.modules:
      module.step_end(**kwargs)


  def train(self):
    print('\n==========================================\n', flush=True)
    print('STARTING TRAINING!', flush=True)
    print('\n==========================================\n', flush=True)

    utils.garbage_collect()

    self.train_start()

    for epoch in range(self.config.epochs):
      self.epoch_start(epoch)

      for step, batch in enumerate(self.cached_dataloader_train):
        self.step_start(epoch, step)
        self.step(step, batch)
        self.step_end(epoch, step, batch)
        del batch
        if self._stop:
          break
      if self._stop:
        break

      if self.config.shuffle_dataset_each_epoch:
        self.raw_dataset_train.shuffle()

      utils.garbage_collect()
      self.epoch_end(epoch)

    self.train_end()

    self.accelerator.end_training()


  def stop(self):
    self._stop = True


  def step(self, step, batch):
    with self.accelerator.accumulate(self.unet), self.accelerator.accumulate(self.text_encoder):
      loss = self.calc_loss_fn(step, batch, self.unet, self.text_encoder)

      self.accelerator.backward(loss)
      if self.accelerator.sync_gradients:
        self.accelerator.clip_grad_norm_(itertools.chain(self.unet.parameters(), self.text_encoder.parameters()), 1.0)

      self.unet_optimizer.step()
      self.te_optimizer.step()

      self.unet_lr_scheduler.step()
      self.te_lr_scheduler.step()

      self.unet_optimizer.zero_grad(set_to_none=True)
      self.te_optimizer.zero_grad(set_to_none=True)

      self.last_loss = loss.item()


  def calc_loss_fn(self, step, batch, unet, text_encoder, mode='train'):
    noise_pred, ground_truth, timestep = train_utils.get_unet_pred_ground_truth(
      batch=batch,
      unet=unet,
      text_encoder=text_encoder,
      scheduler=self.scheduler,
      clip_penultimate=self.config.clip_penultimate,
      offset_noise_weight=self.config.offset_noise_weight,
    )

    loss, loss_with_snr = train_utils.calc_unet_loss(
      step=step,
      noise_pred=noise_pred,
      ground_truth=ground_truth,
      timestep=timestep,
      snr=self.config.snr,
      use_snr=self.config.use_snr,
      scheduler=self.scheduler,
      snr_warmup_steps=self.config.snr_warmup_steps,
    )

    if mode == 'train':
      return loss_with_snr

    return loss, loss_with_snr

  def setup_run_dir(self):
    self.run_dir = os.path.join('./runs', self.config.run_name)
    if self.accelerator.is_main_process:
      os.makedirs(self.run_dir, exist_ok=True)
      shutil.copyfile(self.config.config_path, os.path.join(self.run_dir, 'config.json5'))
    print('Run directory: ' + self.run_dir, flush=True)


  def create_accelerator(self):
    # noinspection PyArgumentList
    self.accelerator = Accelerator(
      gradient_accumulation_steps=self.config.gradient_accumulation_steps,
      mixed_precision='fp16',
      log_with='tensorboard',
      project_dir=os.path.abspath('./runs'),
    )
    original_print = builtins.print
    def print_fn(*args, ** kwargs):
      if self.accelerator.is_main_process:
        original_print(*args, ** kwargs)
    builtins.print = print_fn
    self.config.run_name = broadcast_object_list([self.config.run_name])[0]
    self.config.unet_lr *= self.accelerator.num_processes
    self.config.te_lr *= self.accelerator.num_processes
    self.accelerator.init_trackers(self.config.run_name, config={})
    self.device = self.accelerator.device


  def create_reporter(self):
    self.reporter_config.accelerator = self.accelerator
    self.reporter_config.epochs = self.config.epochs
    self.reporter_config.steps_per_epoch = self.steps_per_epoch
    self.reporter_config.total_steps = self.total_steps
    self.reporter_config.batch_size = self.config.batch_size
    self.reporter_config.run_dir = self.run_dir
    self.reporter = Reporter(self.reporter_config)


  def create_imagen(self):
    self.imagen_config.pretrained_model_name_or_path = self.config.pretrained_model_name_or_path
    self.imagen_config.reporter = self.reporter
    self.imagen_config.vae = self.vae
    self.imagen_config.unet = self.unet
    self.imagen_config.text_encoder = self.text_encoder
    self.imagen_config.tokenizer = self.tokenizer
    self.imagen_config.raw_dataset = self.raw_dataset_train
    self.imagen_config.resolution = self.config.resolution
    self.imagen_config.scheduler = self.imagen_scheduler
    self.imagen_config.accelerator = self.accelerator
    self.imagen = Imagen(self.imagen_config)


  def create_saver(self):
    self.saver_config.checkpoint_name = self.config.run_name
    self.saver_config.pretrained_model_name_or_path = self.config.pretrained_model_name_or_path
    self.saver_config.save_dir = os.path.join(self.run_dir, 'models')
    self.saver_config.vae = self.vae
    self.saver_config.unet = self.unet
    self.saver_config.text_encoder = self.text_encoder
    self.saver_config.tokenizer = self.tokenizer
    self.saver_config.scheduler = self.imagen_scheduler
    self.saver_config.accelerator = self.accelerator
    self.saver = Saver(self.saver_config)



  def create_controller(self):
    self.controller = Controller(self)


  def create_validator(self):
    self.validator = Validator(ValidatorConfig(
      seed=self.config.seed,
      calc_loss_fn=self.calc_loss_fn,
      dataloader=self.cached_dataloader_val,
      reporter=self.reporter,
      unet=self.unet,
      text_encoder=self.text_encoder,
      accelerator=self.accelerator,
      validate_every_n_minutes=self.config.validate_every_n_minutes,
      validate_every_n_epochs=self.config.validate_every_n_epochs,
      validate_every_n_steps=self.config.validate_every_n_steps,
      validate_at_training_start=self.config.validate_at_training_start,
      validate_at_training_end=self.config.validate_at_training_end,
    ))


  def prepare_accelerator(self):
    with utils.Timer('accelerator.prepare'):
      self.unet, self.text_encoder, self.unet_optimizer, self.te_optimizer, self.cached_dataloader_train, self.unet_lr_scheduler, self.te_lr_scheduler = \
        self.accelerator.prepare(self.unet, self.text_encoder, self.unet_optimizer, self.te_optimizer, self.cached_dataloader_train, self.unet_lr_scheduler, self.te_lr_scheduler)
    self.steps_per_epoch = len(self.cached_dataloader_train)
    self.total_steps = self.steps_per_epoch * self.config.epochs


  def load_sd(self):
    sd = sd_utils.load_sd(self.config.pretrained_model_name_or_path, self.device, use_cache=self.config.cache_models)
    self.tokenizer, self.text_encoder, self.unet, self.vae, self.scheduler = sd
    self.imagen_scheduler = EulerDiscreteScheduler.from_pretrained(self.config.pretrained_model_name_or_path, subfolder='scheduler')


  def get_optimizer(self, params_to_optimize, lr):
    if self.config.optimizer == 'adam':
      return bitsandbytes.optim.AdamW8bit(
        params_to_optimize,
        lr=lr,
        betas=(self.config.adam_optimizer_beta_one, self.config.adam_optimizer_beta_two),
        weight_decay=self.config.adam_optimizer_weight_decay
      )
    return Lion(
      params_to_optimize,
      lr=lr,
      betas=(self.config.lion_optimizer_beta_one, self.config.lion_optimizer_beta_two),
      weight_decay=self.config.lion_optimizer_weight_decay,
    )

  def create_optimizer(self):
    self.unet_optimizer = self.get_optimizer(self.unet.parameters(), self.config.unet_lr)
    self.te_optimizer = self.get_optimizer(self.text_encoder.parameters(), self.config.te_lr)


  @staticmethod
  def get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: int = 1, last_epoch: int = -1):
    def lr_lambda(current_step):
      if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
      progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
      if progress >= 1.0:
        return 0.0
      return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


  def get_lr_scheduler(self, name, optimizer, num_warmup_steps, num_training_steps, num_cycles):
    if name != 'cosine_with_restarts':
      return get_lr_scheduler(name=name, optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    return self.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles)


  def create_lr_scheduler(self):
    unet_epochs = self.config.unet_lr_epochs if self.config.unet_lr_epochs != 0 else self.config.epochs
    te_epochs = self.config.te_lr_epochs if self.config.te_lr_epochs != 0 else self.config.epochs

    self.unet_lr_scheduler = self.get_lr_scheduler(
      self.config.unet_lr_scheduler, optimizer=self.unet_optimizer, num_warmup_steps=self.config.unet_lr_warmup_steps,
      num_training_steps=unet_epochs * len(self.cached_dataloader_train),
      num_cycles=self.config.epochs,
    )
    self.te_lr_scheduler = self.get_lr_scheduler(
      self.config.te_lr_scheduler, optimizer=self.te_optimizer, num_warmup_steps=self.config.te_lr_warmup_steps,
      num_training_steps= te_epochs * len(self.cached_dataloader_train),
      num_cycles=self.config.epochs,
    )


  def load_raw_datasets(self):
    train_dataset_config = RawDatasetConfig(
      val_split=self.config.dataset_val_split,
      max_val_images=self.config.dataset_max_val_images,
      directory=self.config.dataset_dir,
      cond_dropout=self.config.cond_dropout,
      shuffle_captions=self.config.shuffle_captions,
      device=self.device,
      resolution=self.config.resolution,
      batch_size=self.config.batch_size,
      seed=self.config.seed,
      type='train',
      accelerator=None,
      max_images=self.config.dataset_max_images,
      ignore_cache=self.config.ignore_cache,
    )
    val_dataset_config = RawDatasetConfig(**dataclasses.asdict(train_dataset_config))
    val_dataset_config.type = 'val'
    val_dataset_config.accelerator = self.accelerator
    train_dataset_config.accelerator = self.accelerator

    self.raw_dataset_train = RawDataset(train_dataset_config)
    self.raw_dataset_val = RawDataset(val_dataset_config)


  def load_cached_datasets(self):
    self.cached_dataset_train = CachedDataset(CachedDatasetConfig(raw_dataset=self.raw_dataset_train, vae=self.vae))
    self.cached_dataset_val = CachedDataset(CachedDatasetConfig(raw_dataset=self.raw_dataset_val, vae=self.vae))


  def load_cached_dataloaders(self):
    collate_fn = CachedDataset.make_collate_fn(self.tokenizer, self.device)

    self.cached_dataloader_train = DataLoader(self.cached_dataset_train, batch_size=self.config.batch_size, shuffle=False, collate_fn=collate_fn)
    self.cached_dataloader_val = DataLoader(self.cached_dataset_val, batch_size=self.config.batch_size, shuffle=False, collate_fn=collate_fn)

