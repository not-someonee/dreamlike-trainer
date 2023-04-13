from Reporter import Reporter
from RawDataset import RawDataset

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel, AutoencoderKL
from accelerate import Accelerator

from dataclasses import dataclass

@dataclass
class GenParams:
  prompt: str = ''
  negative_prompt: str = ''
  seed: int = None
  steps: int = None
  scale: float = None
  aspect_ratio: float = None

@dataclass
class ImagenConfig:
  gen_every_n_minutes: float = 5.0
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

  pretrained_model_name_or_path: str = ''
  reporter: Reporter = None
  vae: AutoencoderKL = None
  unet: UNet2DConditionModel = None
  text_encoder: CLIPTextModel = None
  tokenizer: CLIPTokenizer = None
  raw_dataset: RawDataset = None
  resolution: int = None
  accelerator: Accelerator = None

class Imagen:
  def __init__(self, config: ImagenConfig):
    self.config = config

  def train_start(self):
    pass

  def train_end(self):
    pass

  def epoch_start(self, epoch: int):
    pass

  def epoch_end(self, epoch: int):
    pass

  def step_start(self, epoch: int, step: int):
    pass

  def step_end(self, epoch: int, step: int, batch, loss: float):
    pass