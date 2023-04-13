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


# Decode latents to pytorch tensor
def decode_latents(vae, latents):
  latents = 1 / 0.18215 * latents
  image = vae.decode(latents).sample
  image = (image / 2 + 0.5).clamp(0, 1)
  image = image.cpu().permute(0, 2, 3, 1).float()
  return image


def garbage_collect():
  with Timer('Collecting GC garbage, freeing GPU memory'):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
  print('', flush=True)