from pathlib import Path
from typing import Any

import torch

from hex_tileable_diffusion.config import IPAdapterConfig
import numpy as np
from PIL import Image
from transformers import CLIPImageProcessor

def load_ip_adapter(pipe: Any, config: IPAdapterConfig, cache_dir: str | Path | None = None) -> None:
    kwargs: dict[str, Any] = {}
    if cache_dir is not None: kwargs["cache_dir"] = str(cache_dir)
    pipe.load_ip_adapter(config.model_id, subfolder=config.subfolder, weight_name=config.weight_name, **kwargs)

def encode_ip_adapter_image(pipe: Any, image: Image.Image, device: torch.device, do_cfg: bool) -> list[torch.Tensor] | None:
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if hasattr(pipe, "prepare_ip_adapter_image_embeds"):
        embeds: list[torch.Tensor] = pipe.prepare_ip_adapter_image_embeds(
            ip_adapter_image=image,
            ip_adapter_image_embeds=None,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_cfg,
        )
        return embeds

    if pipe.image_encoder is None: return None


    feat_ext = (
        pipe.feature_extractor
        if pipe.feature_extractor is not None
        else CLIPImageProcessor()
    )
    clip_input = feat_ext(images=image, return_tensors="pt").pixel_values
    clip_input = clip_input.to(device=device, dtype=pipe.image_encoder.dtype)

    with torch.no_grad():
        image_embeds = pipe.image_encoder(clip_input).image_embeds

    if do_cfg:
        negative_embeds = torch.zeros_like(image_embeds)
        image_embeds = torch.cat([negative_embeds, image_embeds])

    return [image_embeds]
