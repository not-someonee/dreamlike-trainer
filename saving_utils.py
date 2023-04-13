from transformers import CLIPTextModel, CLIPModel, CLIPTokenizer, CLIPImageProcessor
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, StableDiffusionPipeline, DDIMScheduler

def save_diffusers(save_path: str, tokenizer: CLIPTokenizer, text_encoder: CLIPTextModel, vae: AutoencoderKL, unet: UNet2DConditionModel, torch_dtype, use_safetensors: bool = True):
  pipe = StableDiffusionPipeline(
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    vae=vae,
    unet=unet,
    scheduler=DDIMScheduler.from_pretrained(self.config.pretrained_model_name_or_path, subfolder='scheduler'),
    torch_dtype=torch_dtype,
    requires_safety_checker=False,
    safety_checker=None,
    feature_extractor=None,
  )
  pipe.save_pretrained(save_path, safe_serialization=use_safetensors)
  print('Saved to ' + save_path, flush=True)
