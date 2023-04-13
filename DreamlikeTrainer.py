import setup_env

import utils
from RawDataset import RawDataset, RawDatasetConfig
from CachedDataset import CachedDataset, CachedDatasetConfig
from Reporter import ReporterConfig, Reporter
from Imagen import ImagenConfig, Imagen
from Saver import SaverConfig, Saver
from Controller import Controller
from AestheticsPredictor import AestheticsPredictor
import saving_utils
import train_utils
import sd_utils

from typing import Literal
import dataclasses
from dataclasses import dataclass
import gc
import os
import itertools
import shutil
import io
from datetime import datetime

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from transformers import CLIPTextModel, CLIPModel, CLIPTokenizer, CLIPImageProcessor
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, StableDiffusionPipeline, DDIMScheduler
from diffusers.optimization import get_scheduler as get_lr_scheduler

from accelerate import Accelerator
from accelerate.utils import set_seed

import bitsandbytes


@dataclass
class DreamlikeTrainerConfig:
  pretrained_model_name_or_path: str
  dataset_dir: str
  epochs: int = 4
  batch_size: int = 3
  lr: float = 1e-6
  lr_scheduler: Literal['constant', 'cosine'] = 'constant'
  lr_warmup_steps: int = 0
  seed: int = 42
  clip_penultimate: bool = False
  cond_dropout: float = 0.1
  shuffle_captions: bool = True
  offset_noise_weight: float = 0.07
  resolution: int = 768
  precache_latents: bool = False
  ignore_cache: bool = False
  dataset_val_split: float = 0.1
  dataset_max_val_images: int = 500
  project_dir: str = None


class DreamlikeTrainer:
  config: DreamlikeTrainer = None
  run_name: str = None
  project_name: str = None
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

  accelerator: Accelerator = None
  device = None
  lr_scheduler = None
  optimizer: bitsandbytes.optim.AdamW8bit = None

  raw_dataset_train: RawDataset = None
  raw_dataset_val: RawDataset = None

  cached_dataset_train: CachedDataset = None
  cached_dataset_val: CachedDataset = None
  cached_dataloader_train: DataLoader = None
  cached_dataloader_val: DataLoader = None

  reporter_config: ReporterConfig = None
  imagen_conifg: ImagenConfig = None

  reporter: Reporter = None
  imagen: Imagen = None
  saver: Saver = None
  controller: Controller = None
  modules = List[Reporter, Imagen, Saver] = None

  _stop: bool = False


  def __init__(self, config: DreamlikeTrainerConfig, reporter_config: ReporterConfig, imagen_config: ImagenConfig, saver_config: SaverConfig):
    self.config = config
    self.reporter_config = reporter_config
    self.imagen_conifg = imagen_config
    self.saver_config = saver_config

    self.setup_run_dir()
    self.create_accelerator()

    self.load_sd()

    set_seed(config.seed)

    self.load_raw_datasets()
    self.load_cached_datasets()
    self.load_cached_dataloaders()

    self.create_optimizer()
    self.create_lr_scheduler()

    self.prepare_accelerator()

    self.create_reporter()
    self.create_imagen()
    self.create_saver()
    self.create_controller()

    self.modules = [self.reporter, self.imagen, self.saver, self.controller]


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
    lr = self.lr_scheduler.get_last_lr()[0]
    kwargs = { 'epoch': epoch, 'step': step, 'batch': batch, 'loss': self.last_loss, 'lr': lr, 'global_step': self.global_step }
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

      self.epoch_end(epoch)

    self.train_end()

    self.accelerator.end_training()


  def stop(self):
    self._stop = True


  def step(self, step, batch):
    with self.accelerator.accumulate(self.unet), self.accelerator.accumulate(self.text_encoder):
      noise_pred, ground_truth = train_utils.get_unet_pred_ground_truth(
        batch=batch,
        unet=self.unet,
        text_encoder=self.text_encoder,
        scheduler=self.scheduler,
        clip_penultimate=self.config.clip_penultimate,
        offset_noise_weight=self.config.offset_noise_weight,
      )

      loss = F.mse_loss(noise_pred.float(), ground_truth.float(), reduction='mean')

      self.accelerator.backward(loss)
      if self.accelerator.sync_gradients:
        self.accelerator.clip_grad_norm_(itertools.chain(self.unet.parameters(), self.text_encoder.parameters()), 1.0)
      self.optimizer.step()
      self.lr_scheduler.step()
      self.optimizer.zero_grad()
      self.last_loss = loss.item()


  def setup_run_dir(self):
    self.run_name = datetime.now().strftime('run_%Y-%m-%d_%H-%M-%S')
    self.project_name = os.path.basename(self.config.project_dir)
    self.run_dir = os.path.join(self.config.project_dir, 'runs', run_name)
    os.makedirs(os.path.join(self.config.project_dir, 'runs'), exist_ok=True)
    os.makedirs(self.run_dir)
    shutil.copyfile(os.path.join(self.config.project_dir, 'config.json5'), os.path.join(self.run_dir, 'config.json5'))
    print('Run directory: ' + self.run_dir, flush=True)


  def create_accelerator(self):
    # noinspection PyArgumentList
    self.accelerator = Accelerator(
      gradient_accumulation_steps=1,
      mixed_precision='fp16',
      log_with='tensorboard',
      project_dir=os.path.join(self.config.project_dir, 'runs'),
    )
    self.accelerator.init_trackers(run_name, config={})
    self.device = self.accelerator.device


  def create_reporter(self):
    self.reporter_config.accelerator = self.accelerator
    self.reporter = Reporter(reporter_config)


  def create_imagen(self):
    self.imagen_config.pretrained_model_name_or_path = self.config.pretrained_model_name_or_path
    self.imagen_config.reporter = self.reporter
    self.imagen_config.vae = self.vae
    self.imagen_config.unet = self.unet
    self.imagen_config.text_encoder = self.text_encoder
    self.imagen_config.tokenizer = self.tokenizer
    self.imagen_config.raw_dataset = self.raw_dataset
    self.imagen_config.resolution = self.config.resolution
    self.imagen_config.accelerator = self.accelerator
    self.imagen = Imagen(self.imagen_config)


  def create_saver(self):
    self.saver_config.checkpoint_name = self.project_name
    self.saver_config.save_dir = os.path.join(self.run_dir, 'models')
    self.saver_config.vae = self.vae
    self.saver_config.unet = self.unet
    self.saver_config.text_encoder = self.text_encoder
    self.saver_config.tokenizer = self.tokenizer
    self.saver_config.scheduler = DDIMScheduler.from_pretrained(self.config.pretrained_model_name_or_path, subfolder='scheduler')
    self.saver_config.accelerator = self.accelerator
    self.saver = Saver(self.saver_config)



  def create_controller(self):
    self.controller = Controller(self)


  def prepare_accelerator(self):
    with utils.Timer('accelerator.prepare'):
      self.unet, self.text_encoder, self.optimizer, self.cached_dataloader_train, self.lr_scheduler = \
        self.accelerator.prepare(self.unet, self.text_encoder, self.optimizer, self.cached_dataloader_train, self.lr_scheduler)


  def load_sd(self):
    sd = sd_utils.load_sd(self.config.pretrained_model_name_or_path, self.device, use_cache=True)
    self.tokenizer, self.text_encoder, self.unet, self.vae, self.scheduler = sd


  def create_optimizer(self):
    params_to_optimize = itertools.chain(self.unet.parameters(), self.text_encoder.parameters())
    self.optimizer = bitsandbytes.optim.AdamW8bit(params_to_optimize, lr=self.config.lr, betas=(0.9, 0.999), weight_decay=1e-2)


  def create_lr_scheduler(self):
    self.lr_scheduler = get_lr_scheduler(
      self.config.lr_scheduler, optimizer=self.optimizer, num_warmup_steps=self.config.lr_warmup_steps,
      num_training_steps=self.config.epochs * len(self.cached_dataloader_train)
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
      ignore_cache=self.config.ignore_cache,
    )
    val_dataset_config = RawDatasetConfig(**dataclasses.asdict(train_dataset_config))
    val_dataset_config.type = 'val'

    self.raw_dataset_train = RawDataset(train_dataset_config)
    self.raw_dataset_val = RawDataset(val_dataset_config)


  def load_cached_datasets(self):
    self.cached_dataset_train = CachedDataset(CachedDatasetConfig(raw_dataset=self.raw_dataset_train, vae=self.vae))
    self.cached_dataset_val = CachedDataset(CachedDatasetConfig(raw_dataset=self.raw_dataset_val, vae=self.vae))


  def load_cached_dataloaders(self):
    collate_fn = CachedDataset.make_collate_fn(self.tokenizer, self.device)

    self.cached_dataloader_train = DataLoader(self.cached_dataset_train, batch_size=self.config.batch_size, shuffle=False, collate_fn=collate_fn)
    self.cached_dataloader_val = DataLoader(self.cached_dataset_val, batch_size=self.config.batch_size, shuffle=False, collate_fn=collate_fn)

    self.steps_per_epoch = len(self.cached_dataloader_train)
    self.total_steps = steps_per_epoch * self.config.epochs

