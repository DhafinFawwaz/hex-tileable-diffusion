import os
from typing import Any

import torch
from PIL import Image

from hex_tileable_diffusion.config import (
    ControlNetConfig,
    DiffusionConfig,
    ExteriorPassConfig,
    FinetuneConfig,
    IPAdapterConfig,
    PostprocessConfig,
)
from hex_tileable_diffusion.conditioning.controlnet import load_controlnet
from hex_tileable_diffusion.conditioning.ip_adapter import (
    encode_ip_adapter_image,
    load_ip_adapter,
)
from hex_tileable_diffusion.core.hexwrapper import HexWrapper
from hex_tileable_diffusion.diffusion.rolling_inpaint import run_rolling_inpaint
from hex_tileable_diffusion.diffusion.scheduling import create_scheduler
from hex_tileable_diffusion.observer.hexobserver import HexObserver
import numpy as np

class HexInpaintPipeline:

    pipe: Any
    controlnet: Any
    ip_adapter_embeds: list[torch.Tensor] | None
    device: torch.device

    def __init__(
        self,
        diffusion_config: DiffusionConfig,
        controlnet_config: ControlNetConfig = ControlNetConfig(),
        ip_adapter_config: IPAdapterConfig = IPAdapterConfig(),
        cache_dir: str | None = ".cache",
    ) -> None:
        self._diffusion_config = diffusion_config
        self._controlnet_config = controlnet_config
        self._ip_adapter_config = ip_adapter_config
        self._cache_dir = cache_dir

        self.pipe = None
        self.controlnet = None
        self.ip_adapter_embeds = None
        self._ip_adapter_model_id: str | None = None
        self._ip_adapter_scale: float | None = None
        self._scheduler_name = ""
        self._controlnet_model_id = controlnet_config.model_id if controlnet_config.enabled else None
        self.device = torch.device("cpu")


    def download_or_get_from_cache(self) -> None:
        dc = self._diffusion_config
        cache_dir = self._cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        from diffusers import StableDiffusionInpaintPipeline  # type: ignore[import-not-found]

        # VAE
        _vae_kwargs: dict[str, Any] = {}
        if dc.vae_model:
            from diffusers import AutoencoderKL  # type: ignore[import-not-found,unused-ignore]

            custom_vae = AutoencoderKL.from_pretrained(
                dc.vae_model,
                torch_dtype=torch.float16,
                cache_dir=cache_dir,
            )
            _vae_kwargs["vae"] = custom_vae

        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            dc.model_id,
            torch_dtype=torch.float16,
            safety_checker=None,
            cache_dir=cache_dir,
            **_vae_kwargs,
        )
        self.pipe = self.pipe.to("cuda")
        self.device = self.pipe.device

        # Scheduler
        self._scheduler_name = ""
        if dc.scheduler_type is not None:
            orig_name = type(self.pipe.scheduler).__name__
            self.pipe.scheduler = create_scheduler(dc.scheduler_type, dict(self.pipe.scheduler.config))
            self._scheduler_name = (f"{orig_name}: {type(self.pipe.scheduler).__name__}")

        # ControlNet
        self.controlnet = load_controlnet(self._controlnet_config, cache_dir=cache_dir)

        ip_cfg = self._ip_adapter_config
        if ip_cfg.enabled and ip_cfg.model_id:
            load_ip_adapter(self.pipe, ip_cfg, cache_dir=cache_dir)
            self.pipe.set_ip_adapter_scale(ip_cfg.scale)
            self._ip_adapter_model_id = ip_cfg.model_id
            self._ip_adapter_scale = ip_cfg.scale



    def encode_ip_reference(
        self,
        image: Image.Image,
        guidance_scale: float,
    ) -> None:
        do_cfg = guidance_scale > 1.0
        self.ip_adapter_embeds = encode_ip_adapter_image(self.pipe, image, self.device, do_cfg)

    def inpaint(
        self,
        source_image: np.ndarray,
        mask_image: np.ndarray,
        prompt: str,
        negative_prompt: str,
        *,
        gen_size: tuple[int, int],
        wrapper: HexWrapper,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        strength: float | None = None,
        seed: int | None = None,
        use_rolling_noise: bool | None = None,
        control_image: np.ndarray | None = None,
        use_controlnet: bool = True,
        use_latent_color_correction: bool = False,
        observer: HexObserver | None = None,
        output_dir: str = ".",
    ) -> np.ndarray:
        dc = self._diffusion_config
        steps = num_inference_steps if num_inference_steps is not None else dc.num_inference_steps
        gs = guidance_scale if guidance_scale is not None else dc.guidance_scale
        st = strength if strength is not None else dc.strength
        sd = seed if seed is not None else dc.seed
        rolling = use_rolling_noise if use_rolling_noise is not None else dc.use_rolling_noise

        cn = self.controlnet if use_controlnet else None

        result = run_rolling_inpaint(
            pipe=self.pipe,
            source_image=source_image,
            mask_image=mask_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=gs,
            strength=st,
            seed=sd,
            gen_size=gen_size,
            wrapper=wrapper,
            use_rolling_noise=rolling,
            roll_mode=dc.roll_mode,
            guidance_schedule=dc.guidance_schedule,
            controlnet=cn,
            control_image=control_image,
            controlnet_conditioning_scale=self._controlnet_config.conditioning_scale,
            ip_adapter_image_embeds=self.ip_adapter_embeds,
            use_latent_color_correction=use_latent_color_correction,
            vae_fp32=dc.vae_fp32,
            observer=observer,
            output_dir=output_dir,
        )

        return result
