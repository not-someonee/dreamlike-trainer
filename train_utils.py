import torch
from transformers import CLIPTextModel
from diffusers import DDPMScheduler, UNet2DConditionModel

# Get predicted noise and ground truth (noise or noise velocity depending on SD version)
def get_unet_pred_ground_truth(clip_penultimate: bool, offset_noise_weight: float, unet: UNet2DConditionModel, text_encoder: CLIPTextModel, batch, scheduler: DDPMScheduler):
  device = unet.device
  latents = batch['latents']
  caption_token_ids = batch['caption_token_ids']
  batch_size = latents.shape[0]

  noise = torch.randn_like(latents)
  # Offset noise (https://www.crosslabs.org/blog/diffusion-with-offset-noise)
  noise += (offset_noise_weight * torch.randn(latents.shape[0], latents.shape[1], 1, 1).to(device))
  noise += 0.1 * torch.randn_like(noise)

  # Sample a random timestep for each image
  timestep = torch.randint(0, scheduler.config.num_train_timesteps, (batch_size,), device=device).long()

  # Add noise to the latents according to the noise magnitude at each timestep
  # (this is the forward diffusion process)
  noisy_latents = scheduler.add_noise(latents, noise, timestep)

  if clip_penultimate:
    encoder_hidden_states = text_encoder(caption_token_ids, output_hidden_states=True)
    encoder_hidden_states = text_encoder.text_model.final_layer_norm(encoder_hidden_states['hidden_states'][-2])
  else:
    encoder_hidden_states = text_encoder(caption_token_ids)[0]

  # Noise predicted by the unet
  noise_pred = unet(noisy_latents, timestep, encoder_hidden_states).sample

  # Ground truth noise to compute the loss against (noise for SD 1.x, noise velocity thingy for SD 2.x)
  if scheduler.config.prediction_type == 'epsilon':
    ground_truth = noise
  elif scheduler.config.prediction_type == 'v_prediction':
    ground_truth = scheduler.get_velocity(latents, noise, timestep)
  else:
    raise Exception('Unsupported scheduler.config.prediction_type: ' + scheduler.config.prediction_type)

  return noise_pred, ground_truth