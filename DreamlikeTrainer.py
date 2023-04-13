import utils
from RawDataset import RawDataset, RawDatasetConfig
from CachedDataset import CachedDataset, CachedDatasetConfig
from Reporter import ReporterConfig, Reporter
from Imagen import ImagenConfig, Imagen
from AestheticsPredictor import AestheticsPredictor
import saving_utils
import train_utils

from typing import Literal
from dataclasses import dataclass
import gc
import os
import itertools
import warnings
import shutil
import io
import contextlib
from datetime import datetime

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from transformers import CLIPTextModel, CLIPModel, CLIPTokenizer, CLIPImageProcessor
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, StableDiffusionPipeline, DDIMScheduler
from diffusers.optimization import get_scheduler as get_lr_scheduler

from accelerate import Accelerator
from accelerate.utils import set_seed

# Supress bitsandbytes warnings
with contextlib.redirect_stdout(io.StringIO()):
  import bitsandbytes

warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
# Enable fast safetensors load to gpu
os.environ['SAFETENSORS_FAST_GPU'] = '1'

torch.backends.cudnn.benchmark = False


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
  run_dir: str = None

class DreamlikeTrainer:
  def __init__(self, config: DreamlikeTrainerConfig, reporter_config: ReporterConfig, imagen_config: ImagenConfig):
    self.config = config
    self.config.dataset_dir = os.path.abspath(self.config.dataset_dir)
    run_name = datetime.now().strftime('run_%Y-%m-%d_%H-%M-%S')
    self.config.run_dir = os.path.join(self.config.project_dir, 'runs', run_name)
    os.makedirs(os.path.join(self.config.project_dir, 'runs'), exist_ok=True)
    os.makedirs(self.config.run_dir)
    shutil.copyfile(os.path.join(self.config.project_dir, 'config.json5'), os.path.join(self.config.run_dir, 'config.json5'))

    print('Run directory: ' + self.config.run_dir, flush=True)

    # noinspection PyArgumentList
    self.accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision='fp16',
        log_with='tensorboard',
        project_dir=os.path.join(self.config.project_dir, 'runs'),
    )
    self.accelerator.init_trackers(run_name, config={})
    self.device = self.accelerator.device

    set_seed(config.seed)

    with utils.Timer('Loading SD'):
      self.tokenizer, self.text_encoder, self.vae, self.unet, self.scheduler = self.load_sd(config, self.device)
    print('', flush=True)

    self.raw_dataset = self.load_raw_dataset(config, self.tokenizer, self.device)
    print('', flush=True)
    self.train_dataset, self.train_dataloader = self.load_cached_dataset_dataloader(config, self.tokenizer, self.raw_dataset, self.vae, self.device)
    print('', flush=True)

    reporter_config.accelerator = self.accelerator
    self.reporter = Reporter(reporter_config)
    imagen_config.unet = self.unet
    imagen_config.pretrained_model_name_or_path = config.pretrained_model_name_or_path
    imagen_config.reporter = self.reporter
    imagen_config.vae = self.vae
    imagen_config.unet = self.unet
    imagen_config.text_encoder = self.text_encoder
    imagen_config.tokenizer = self.tokenizer
    imagen_config.raw_dataset = self.raw_dataset
    imagen_config.resolution = self.config.resolution
    imagen_config.accelerator = self.accelerator
    self.imagen = Imagen(imagen_config)

    self.optimizer = self.create_optimizer(config, self.unet, self.text_encoder)
    self.lr_scheduler = self.create_lr_scheduler(config, self.optimizer, self.train_dataloader)

    with utils.Timer('accelerator.prepare'):
      self.unet, self.text_encoder, self.optimizer, self.train_dataloader, self.lr_scheduler = \
        self.accelerator.prepare(self.unet, self.text_encoder, self.optimizer, self.train_dataloader, self.lr_scheduler)


  def train(self):
    print('', flush=True)
    print('==========================================', flush=True)
    print('', flush=True)
    print('STARTING TRAINING!', flush=True)
    print('', flush=True)
    print('==========================================', flush=True)
    print('', flush=True)

    utils.garbage_collect()

    steps_per_epoch = len(self.train_dataloader)
    total_steps = steps_per_epoch * self.config.epochs

    self.reporter.init(self, steps_per_epoch, total_steps)
    self.reporter.train_start()
    self.imagen.train_start()

    for epoch in range(self.config.epochs):
      self.reporter.epoch_start(epoch)
      self.imagen.epoch_start(epoch)
      self.train_epoch(epoch)
      self.reporter.epoch_end(epoch)
      self.imagen.epoch_end(epoch)

      if self.config.ignore_cache:
        print('Recalculated cache, switching ignore_cache to False')
        self.config.ignore_cache = False
        self.raw_dataset.config.ignore_cache = False

      print('', flush=True)
      print('', flush=True)
      save_path = os.path.join(self.config.run_dir, 'ep_' + str(epoch) + '__' + str(epoch * steps_per_epoch + steps_per_epoch) + 's')
      print('Saving to ' + save_path + '...', flush=True)
      saving_utils.save_diffusers(
        save_path=save_path,
        tokenizer=self.tokenizer,
        text_encoder=self.accelerator.unwrap_model(self.text_encoder),
        vae=AutoencoderKL.from_pretrained(self.config.pretrained_model_name_or_path, subfolder='vae').to(self.accelerator.device),
        unet=self.accelerator.unwrap_model(self.unet),
        torch_dtype=torch.float16,
      )
      print('', flush=True)
      print('', flush=True)

    self.reporter.train_end()
    self.imagen.train_end()

    self.accelerator.end_training()


  def train_epoch(self, epoch):
    for step, batch in enumerate(self.train_dataloader):
      self.reporter.step_start(epoch, step)
      self.imagen.step_start(epoch, step)
      loss = self.step(step, batch)
      self.reporter.step_end(epoch, step, batch, loss)
      self.imagen.step_end(epoch, step, batch)
      del batch



  def step(self, step, batch):
    with self.accelerator.accumulate(self.unet), self.accelerator.accumulate(self.text_encoder):
      noise_pred, ground_truth = train_utils.get_pred_ground_truth(
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
      return loss.item()


  @staticmethod
  def load_sd(config, device):
    with utils.Timer('Loading tokenizer'):
      tokenizer = CLIPTokenizer.from_pretrained(config.pretrained_model_name_or_path, subfolder='tokenizer')
    with utils.Timer('Loading text_encoder'):
      text_encoder = CLIPTextModel.from_pretrained(config.pretrained_model_name_or_path, subfolder='text_encoder', torch_dtype=torch.float32)
      text_encoder.to(device)
    with utils.Timer('Loading unet'):
      unet = UNet2DConditionModel.from_pretrained(config.pretrained_model_name_or_path, subfolder='unet', torch_dtype=torch.float32)
      unet.enable_xformers_memory_efficient_attention()
      unet.to(device)
    with utils.Timer('Loading vae'):
      vae = AutoencoderKL.from_pretrained(config.pretrained_model_name_or_path, torch_dtype=torch.float16, subfolder='vae')
      vae.enable_xformers_memory_efficient_attention()
      vae.to(device)
      vae.requires_grad_(False)
    scheduler = DDPMScheduler.from_pretrained(config.pretrained_model_name_or_path, subfolder='scheduler')

    return tokenizer, text_encoder, vae, unet, scheduler




  @staticmethod
  def create_optimizer(config, unet, text_encoder):
    params_to_optimize = itertools.chain(unet.parameters(), text_encoder.parameters())
    return bitsandbytes.optim.AdamW8bit(
      params_to_optimize,
      lr=config.lr,
      betas=(0.9, 0.999),
      weight_decay=1e-2,
    )


  @staticmethod
  def create_lr_scheduler(config, optimizer, train_dataloader):
    return get_lr_scheduler(
      config.lr_scheduler,
      optimizer=optimizer,
      num_warmup_steps=config.lr_warmup_steps,
      num_training_steps=config.epochs * len(train_dataloader),
    )


  @staticmethod
  def load_raw_dataset(config, tokenizer, device):
    # Load RawDataset (e.g. it returns image tensors + caption token ids)
    return RawDataset(RawDatasetConfig(
      val_split=config.dataset_val_split,
      max_val_images=config.dataset_max_val_images,
      directory=config.dataset_dir,
      cond_dropout=config.cond_dropout,
      shuffle_captions=config.shuffle_captions,
      device=device,
      resolution=config.resolution,
      batch_size=config.batch_size,
      seed=config.seed,
      type='train',
      ignore_cache=config.ignore_cache,
    ))


  @staticmethod
  def load_cached_dataset_dataloader(config: DreamlikeTrainerConfig, tokenizer: CLIPTokenizer, raw_dataset: RawDataset, vae: AutoencoderKL, device):
    # Load CachedDataset (returns { 'image_latent': ..., 'caption_token_ids': ..., 'megapixels': ... })
    cached_dataset = CachedDataset(CachedDatasetConfig(raw_dataset=raw_dataset, vae=vae))

    @torch.no_grad()
    def collate_fn(batch):
      latents = [v['image_latent'] for v in batch]
      caption_token_ids = [

        torch.tensor(tokenizer(
          v['caption'],
          padding='max_length',
          truncation=True,
          max_length=tokenizer.model_max_length,
        ).input_ids, device=device)

        for v in batch
      ]
      megapixels = sum([v['megapixels'] for v in batch])


      del batch

      latents = torch.stack(latents).float()
      caption_token_ids = torch.stack(caption_token_ids)

      return { 'latents': latents, 'caption_token_ids': caption_token_ids, 'megapixels': megapixels }

    cached_dataloader = DataLoader(
      cached_dataset,
      batch_size=config.batch_size,
      shuffle=False,
      collate_fn=collate_fn,
    )

    return cached_dataset, cached_dataloader

