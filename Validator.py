from Reporter import Reporter
from dataclasses import dataclass
from torch.utils.data import DataLoader


@dataclass
class ValidatorConfig:
  dataloader: Dataloader
  reporter: Reporter


class Validator:
  def __int__(self, config: ValidatorConfig):
    self.config = config

  def validate(self):
    pass