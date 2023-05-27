import math

import torch
import torch.nn.functional as F

from transformers import CLIPTextModel
from diffusers import DDPMScheduler, UNet2DConditionModel

import sd_utils


# https://arxiv.org/pdf/2303.09556.pdf
def get_snr_weight(timesteps, scheduler, gamma):
    alphas_cumprod = scheduler.alphas_cumprod
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    alpha = sqrt_alphas_cumprod
    sigma = sqrt_one_minus_alphas_cumprod
    all_snr = (alpha / sigma) ** 2
    snr = torch.stack([all_snr[t] for t in timesteps])
    gamma_over_snr = torch.div(torch.ones_like(snr) * gamma, snr)
    snr_weight = torch.minimum(gamma_over_snr, torch.ones_like(gamma_over_snr)).float()
    return snr_weight


# Get predicted noise and ground truth (noise or noise velocity depending on SD version)
def get_unet_pred_ground_truth(clip_penultimate: bool, offset_noise_weight: float, unet: UNet2DConditionModel,
                               text_encoder: CLIPTextModel, batch, scheduler: DDPMScheduler, tokenizer, device):
    latents = batch['latents'].to(device)
    caption_token_ids = [sd_utils.tokenize(tokenizer, caption, device) for caption in batch['captions']]
    caption_token_ids = torch.stack(caption_token_ids).to(device)
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

    return noise_pred, ground_truth, timestep


@torch.compile
def calc_unet_loss(step, noise_pred, ground_truth, timestep, use_snr, snr, snr_warmup_steps, scheduler):
    loss_orig = F.mse_loss(noise_pred.float(), ground_truth.float(), reduction='mean')
    loss_orig.to(noise_pred.device)

    if use_snr:
        snr_lerp_factor = 1.0 if step >= snr_warmup_steps else (step / snr_warmup_steps)
        loss = torch.lerp(loss_orig, loss_orig * get_snr_weight(timestep, scheduler, snr).to(noise_pred.device),
                          snr_lerp_factor).mean()
        return loss_orig, loss
    return loss_orig, loss_orig
