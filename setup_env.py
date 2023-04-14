import contextlib
import warnings
import torch
import os
import io

# Supress bitsandbytes warnings
with contextlib.redirect_stdout(io.StringIO()):
  import bitsandbytes

warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
# Enable fast safetensors load to gpu
os.environ['SAFETENSORS_FAST_GPU'] = '1'

torch.backends.cudnn.benchmark = False

