import random
from dataclasses import dataclass
from math import isclose
import time
import tqdm
from typing import List
import traceback

import utils
import keyboard_util
from Reporter import Reporter
from RawDataset import RawDataset
from SDPipeline import SDPipeline

import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel, AutoencoderKL, EulerDiscreteScheduler
from accelerate import Accelerator
from pytorch_lightning.utilities.seed import isolate_rng
import tomesd


@dataclass
class GenParams:
  prompt: str = ''
  negative_prompt: str = ''
  seed: int = None
  steps: int = None
  scale: float = None
  aspect_ratio: float = None
  width: int = None
  height: int = None


@dataclass
class ImagenConfig:
  gen: bool = True
  gen_every_n_minutes: float = 5.0
  gen_every_n_steps: int = 5
  gen_every_n_epochs: int = 1
  gen_on_training_start: bool = True
  gen_on_training_end: bool = True
  num_gens_from_dataset: int = 10
  steps: int = 25
  scale: float = 7.5
  seed: int = -1
  aspect_ratio: float = 0.75
  prompt_prepend: str = ''
  negative_prompt_prepend: str = ''
  gens: List = None

  tome_ratio: float = 0.25

  pretrained_model_name_or_path: str = ''
  reporter: Reporter = None
  vae: AutoencoderKL = None
  scheduler: EulerDiscreteScheduler = None
  unet: UNet2DConditionModel = None
  text_encoder: CLIPTextModel = None
  tokenizer: CLIPTokenizer = None
  raw_dataset: RawDataset = None
  resolution: int = None
  accelerator: Accelerator = None


class Imagen:
  pipe = None
  gen_id: int = 0
  global_step: int = 0
  use_tome: bool = None

  def __init__(self, config: ImagenConfig):
    self.config = config
    self.use_tome = not isclose(config.tome_ratio, 0.0, abs_tol=0.001)
    self.last_gen_at = time.time()

  def train_start(self):
    if self.config.gen_on_training_start:
      self.gen()

  def train_end(self):
    if self.config.gen_on_training_end:
      self.gen()

  def epoch_start(self, epoch: int):
    pass

  def epoch_end(self, epoch: int):
    if epoch == 0:
      if self.config.gen_every_n_epochs == 1:
        self.gen()
    elif (epoch % self.config.gen_every_n_epochs) == 0:
      self.gen()

  def step_start(self, epoch: int, step: int):
    pass

  def step_end(self, epoch: int, step: int, global_step: int, unet_lr: float, te_lr: float, batch, loss: float):
    self.global_step = global_step
    if (time.time() - self.last_gen_at) > (self.config.gen_every_n_minutes * 60.0):
      self.gen()
    if global_step != 0 and (global_step % self.config.gen_every_n_steps) == 0:
      self.gen()


  @torch.no_grad()
  def prepare_for_gen(self):
    with utils.Timer('Preparing for gen'):
      self.pipe = SDPipeline(
        tokenizer=self.config.tokenizer,
        text_encoder=self.config.accelerator.unwrap_model(self.config.text_encoder),
        vae=AutoencoderKL.from_pretrained(self.config.pretrained_model_name_or_path, subfolder='vae').to(self.config.accelerator.device),
        unet=self.config.accelerator.unwrap_model(self.config.unet),
        scheduler=self.config.scheduler,
        requires_safety_checker=False,
        safety_checker=None,
        feature_extractor=None,
      )

      if self.use_tome:
        tomesd.apply_patch(self.pipe.unet, ratio=self.config.tome_ratio)


  @torch.no_grad()
  def prepare_for_train(self):
    with utils.Timer('Preparing for train'):
      if self.use_tome:
        tomesd.remove_patch(self.pipe.unet)
      self.pipe = None


  def get_gens_from_dataset(self):
    if self.config.num_gens_from_dataset == 0:
      return []

    with isolate_rng():
      random.seed(self.config.seed + self.global_step + 1)
      gens = []
      for i in range(self.config.num_gens_from_dataset):
        gen = GenParams(prompt=self.config.raw_dataset[random.randint(0, len(self.config.raw_dataset) - 1)].caption)
        gens.append(gen)
    return gens


  def get_gens_from_config(self):
    if self.config.gens is None or len(self.config.gens) == 0:
      return []
    return [GenParams(**gen) for gen in self.config.gens]


  def set_default_gen_params(self, gens: List[GenParams]):
    with isolate_rng():
      random.seed(self.config.seed + self.global_step)
      for gen in gens:
        if gen.seed is None or gen.seed == -1:
          gen.seed = self.config.seed
        if gen.seed is None or gen.seed == -1:
          gen.seed = random.randint(0, 9999999)
        if gen.scale is None:
          gen.scale = self.config.scale
        if gen.steps is None:
          gen.steps = self.config.steps
        if gen.aspect_ratio is None:
          gen.aspect_ratio = self.config.aspect_ratio
        if isclose(gen.aspect_ratio, -1, abs_tol=0.001):
          gen.aspect_ratio = random.uniform(0.5, 2)
        if gen.prompt is None:
          gen.prompt = ''
        if gen.negative_prompt is None:
          gen.negative_prompt = ''
        gen.prompt = self.config.prompt_prepend + gen.prompt
        gen.negative_prompt = self.config.negative_prompt_prepend + gen.negative_prompt
        gen.width, gen.height = get_closest_res(gen.aspect_ratio, self.config.resolution)


  @torch.no_grad()
  def gen(self):
    print('\n\n\n', flush=True)
    with utils.Timer('Generating images'):
      self.prepare_for_gen()
      try:
        gens = self.get_gens_from_dataset()
        gens.extend(self.get_gens_from_config())
        self.set_default_gen_params(gens)
        total_steps = 0
        for gen in gens:
          total_steps += gen.steps
        with tqdm.tqdm(total=total_steps * 2, desc='Imagen steps', unit=' steps') as progress_bar:
          images = []
          for i, gen in enumerate(gens):
            progress_bar.desc = f'{i}/{len(gens)} imgs done | Press C to cancel'
            image = self.gen_img(gen, progress_bar)
            images.append(image)
            keys = keyboard_util.get_pressed_keys()
            if 'c' in keys:
              break
          self.config.reporter.report_images(images, '', self.gen_id, gens)
        print('\n', flush=True)
        print('\n', flush=True)
        print('\n', flush=True)
      except Exception:
        traceback.print_exc()
      finally:
        self.prepare_for_train()
        self.gen_id += 1
        self.last_gen_at = time.time()


  @torch.no_grad()
  def gen_img(self, params: GenParams, progress_bar):
    with isolate_rng():
      torch.manual_seed(params.seed)
      def callback(_1, _2, _3):
        progress_bar.update(1)
      pil_images, _ = self.pipe(
        prompt=params.prompt,
        negative_prompt=params.negative_prompt,
        guidance_scale=params.scale,
        num_inference_steps=params.steps,
        width=params.width,
        height=params.height,
        return_dict=False,
        callback=callback,
      )
      return pil_images[0]


# Blame ChatGPT if it doesn't work
def get_closest_res(aspect_ratio: float = 0.75, target_resolution: int = 768, multiple: int = 64) -> (int, int):
  min_diff_resolution = float('inf')
  closest_width = 0
  closest_height = 0

  target_pixel_count = target_resolution * target_resolution
  max_size = int((target_pixel_count * 2) ** 0.5)

  for height in range(multiple, max_size, multiple):
    width = round(height * aspect_ratio / multiple) * multiple
    current_resolution = width * height
    diff_resolution = abs(target_pixel_count - current_resolution)

    if diff_resolution < min_diff_resolution:
      min_diff_resolution = diff_resolution
      closest_width = width
      closest_height = height

  return closest_width, closest_height