import numpy as np
import torch
from diffusers.models import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor


def decode_latents_to_image(vae: AutoencoderKL, image_processor: VaeImageProcessor, latents: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        decoded = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
    return np.array(image_processor.postprocess(decoded, output_type="pil")[0])
