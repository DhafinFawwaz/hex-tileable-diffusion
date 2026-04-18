from pathlib import Path

import torch

from hex_tileable_diffusion.config import ControlNetConfig
from diffusers import ControlNetModel

def load_controlnet(config: ControlNetConfig, device: torch.device | str = "cuda", dtype: torch.dtype = torch.float16, cache_dir: str | Path | None = None) -> ControlNetModel:
    model = ControlNetModel.from_pretrained(config.model_id, torch_dtype=dtype, cache_dir=cache_dir).to(device)
    return model
