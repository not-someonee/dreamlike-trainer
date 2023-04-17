import time
import gc
import torch

class Timer:
  def __init__(self, text):
    self.text = text

  def __enter__(self):
    self.start = time.time()
    print(self.text, flush=True)
    return self

  def __exit__(self, exc_type, exc_value, tb):
    end = time.time()
    print(self.text + ' done, took ' + "{:.2f}".format(end - self.start) + 's', flush=True)


def garbage_collect():
  gc.collect()
  torch.cuda.empty_cache()
  torch.cuda.ipc_collect()