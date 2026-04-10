from typing import Literal, TypeAlias, Union
from PIL import Image
import numpy as np
import torch

ImageInput: TypeAlias = Union[str, Image.Image, np.ndarray, torch.Tensor]

NormOffsetList: TypeAlias = list[tuple[float, float]]

RollMode: TypeAlias = Literal[
    "square",
    "hex",
    "hex_copy",
    "hex_copy_no_roll",
]

SchedulerType: TypeAlias = Literal[
    "euler",
    "euler_a",
    "dpm++_2m",
    "dpm++_2m_karras",
    "dpm++_sde_karras",
    "ddim",
    "uni_pc",
]

MaskType: TypeAlias = Literal["rect", "cross", "border", "strips"]

InpaintModelId: TypeAlias = Literal[
    "runwayml/stable-diffusion-inpainting",
    "stable-diffusion-v1-5/stable-diffusion-inpainting",
    "botp/stable-diffusion-v1-5-inpainting",
    "stabilityai/stable-diffusion-2-inpainting",
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    "Uminosachi/realisticVisionV51_v51VAE-inpainting",
    "Lykon/dreamshaper-8-inpainting",
]

VaeModelId: TypeAlias = Literal[
    "stabilityai/sd-vae-ft-mse",
    "stabilityai/sd-vae-ft-ema",
    "madebyollin/sdxl-vae-fp16-fix",
    "stabilityai/sdxl-vae",
]

ControlNetModelId: TypeAlias = Literal[
    "lllyasviel/control_v11f1e_sd15_tile",
    "lllyasviel/control_v11p_sd15_canny",
    "lllyasviel/control_v11f1p_sd15_depth",
    "lllyasviel/control_v11p_sd15_normalbae",
    "lllyasviel/control_v11p_sd15_lineart",
    "lllyasviel/control_v11p_sd15_softedge",
    "xinsir/controlnet-tile-sdxl-1.0",
    "TTPlanet/TTPLanet_SDXL_Controlnet_Tile_Realistic",
]

IPAdapterModelId: TypeAlias = Literal[
    "h94/IP-Adapter",
    "h94/IP-Adapter-FaceID",
]

IPAdapterWeightName: TypeAlias = Literal[
    "ip-adapter_sd15.bin",
    "ip-adapter_sd15_light.bin",
    "ip-adapter-plus_sd15.bin",
    "ip-adapter-plus-face_sd15.bin",
    "ip-adapter-full-face_sd15.bin",
    "ip-adapter_sdxl.bin",
    "ip-adapter_sdxl_vit-h.bin",
    "ip-adapter-plus_sdxl_vit-h.bin",
]

LogLevel = Literal["debug", "info", "warning", "error"]