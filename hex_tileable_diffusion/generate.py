import os
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from hex_tileable_diffusion.config import HexTileableDiffusionConfig
from hex_tileable_diffusion.observer.hexobserver import HexObserver
from hex_tileable_diffusion.util.image import load_image
from hex_tileable_diffusion.diffusion.pipeline import HexInpaintPipeline
from hex_tileable_diffusion.core.hexwrapper import HexWrapper
from hex_tileable_diffusion.core.hexroller import _in_origin_hex
from hex_tileable_diffusion.core.geometry import _tile_image_hexagonally


@dataclass
class GenerationInfo:
    result: np.ndarray
    R_final: float
    image_arr: np.ndarray
    output_size: int


def generate_hex_tileable_diffusion_texture(config: HexTileableDiffusionConfig, observer: HexObserver | None = None):
    if observer is None: observer = HexObserver()
    observer.on_start()
    observer.preview_count = config.visualization.in_between_preview_count

    torch.cuda.empty_cache()

    if torch.cuda.is_available():
        observer.on_log("info", f"CUDA available: {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB")
    else:
        observer.on_log("warning", "CUDA not available, fallback to CPU. This will be slow!")

    output_path = config.output_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    output_size = config.wrapper.output_size
    observer.on_log("debug", f"Output size: {output_size}x{output_size}")

    image_arr = load_image(config.image_path, output_size)

    # Wrap
    wrapper = HexWrapper(
        img_arr=image_arr,
        hypotenuse=config.wrapper.hypotenuse or output_size / 2,
        x_offset=config.wrapper.x_offset,
        y_offset=config.wrapper.y_offset,
        outer_margin=config.wrapper.outer_margin,
        inner_padding=config.wrapper.inner_padding,
        gap_padding=config.wrapper.gap_padding,
        feather_width=config.wrapper.feather_width,
        horizontal_camera_padding=config.wrapper.horizontal_camera_padding,
        vertical_camera_padding=config.wrapper.vertical_camera_padding,
    )
    rgb_arr, mask_arr = wrapper.wrap()

    observer.on_log("info", "Hexagon image wrapped successfully")
    observer.on_log("debug", "Info", wrapper)

    observer.on_wrapped_finished(wrapper, rgb_arr, mask_arr, config.visualization.hex_outline_thickness)


    hex_pipe = HexInpaintPipeline(
        diffusion_config=config.diffusion,
        controlnet_config=config.controlnet,
        ip_adapter_config=config.ip_adapter,
        cache_dir=config.cache_dir,
    )
    hex_pipe.download_or_get_from_cache()

    observer.on_log("info", "Diffusion pipeline loaded successfully")

    # IP Adapter
    if config.ip_adapter.enabled:
        ref_img = Image.open(str(config.image_path)).convert("RGB")
        hex_pipe.encode_ip_reference(ref_img, config.diffusion.guidance_scale)
        observer.on_log("info", f"IP-Adapter reference encoded from input (scale={config.ip_adapter.scale})")

    # TODO: Finetuning
    if config.exterior.enabled:
        inner_result = _two_pass_inpaint(hex_pipe, wrapper, rgb_arr, mask_arr, image_arr, config, observer)
    else:
        inner_result = _simultaneous_inpaint(hex_pipe, wrapper, rgb_arr, mask_arr, image_arr, config, observer)

    observer.on_after_inpaint(
        rgb_arr=rgb_arr,
        mask_arr=mask_arr,
        result=inner_result,
    )

    # Unwrap
    result, R_final = wrapper.unwrap(inner_result, output_size=output_size)

    observer.on_after_unwrap(wrapper, inner_result, result, R_final, output_size)

    # TODO: Postprocess
    observer.on_after_postprocess(result, result)

    observer.on_log("info", f"Saving final result to {output_path}")
    img = Image.fromarray(result)
    img.save(output_path)

    # _tile_image_hexagonally
    tiled_arr = _tile_image_hexagonally(result, output_size*4, output_size*4, R_final)
    tiled_img = Image.fromarray(tiled_arr.astype(np.uint8))
    tiled_output_path = os.path.splitext(output_path)[0] + "_tiled.png"
    tiled_img.save(tiled_output_path)
    observer.on_log("info", f"Saving tiled result to {tiled_output_path}")

    observer.on_finished(image_arr, result, R_final, output_size)

    return GenerationInfo(result=result, R_final=R_final, image_arr=image_arr, output_size=output_size)


def _simultaneous_inpaint(
    hex_pipe: HexInpaintPipeline,
    wrapper: HexWrapper,
    rgb_arr: np.ndarray,
    mask_arr: np.ndarray,
    image_arr: np.ndarray,
    config: HexTileableDiffusionConfig,
    observer: HexObserver,
) -> Image.Image:
    dc = config.diffusion
    gen_W, gen_H = wrapper.gen_W, wrapper.gen_H
    output_path = config.output_path

    observer.on_log("info", "Starting simultaneous inpaint")

    inner_result = hex_pipe.inpaint(
        source_image=rgb_arr,
        mask_image=mask_arr,
        prompt=dc.prompt,
        negative_prompt=dc.negative_prompt,
        gen_size=(gen_W, gen_H),
        wrapper=wrapper,
        control_image=image_arr,
        use_latent_color_correction=config.postprocess.use_latent_color_correction,
        observer=observer,
        output_dir=output_path,
    )

    observer.on_log("info", "Simultaneous inpaint completed")

    return inner_result


def _two_pass_inpaint(
    hex_pipe: HexInpaintPipeline,
    wrapper: HexWrapper,
    rgb_arr: np.ndarray,
    mask_arr: np.ndarray,
    image_arr: np.ndarray,
    config: HexTileableDiffusionConfig,
    observer: HexObserver,
) -> np.ndarray:
    ext = config.exterior
    dc = config.diffusion
    gen_W, gen_H = wrapper.gen_W, wrapper.gen_H
    ext_seed = ext.seed if ext.seed is not None else dc.seed
    output_path = config.output_path

    # Pass 1: fill gaps, full mask, no rolling, no ControlNet
    observer.on_log("info", "Pass 1: exterior fill (no rolling, no ControlNet)")
    observer.show_denoise_steps = False

    # Temporarily disable IP Adapter for pass 1 if configured
    saved_ip_embeds = None
    if config.ip_adapter.enabled and not config.ip_adapter.use_on_pass1:
        saved_ip_embeds = hex_pipe.ip_adapter_embeds
        hex_pipe.ip_adapter_embeds = None

    pass1 = hex_pipe.inpaint(
        source_image=rgb_arr,
        mask_image=mask_arr,
        prompt=dc.prompt,
        negative_prompt=dc.negative_prompt,
        gen_size=(gen_W, gen_H),
        wrapper=wrapper,
        num_inference_steps=ext.steps,
        guidance_scale=ext.guidance_scale,
        strength=ext.strength,
        seed=ext_seed,
        use_rolling_noise=False,
        use_controlnet=False,
        use_latent_color_correction=config.postprocess.use_latent_color_correction,
        observer=observer,
        output_dir=output_path,
    )

    # Restore IP Adapter embeds
    if saved_ip_embeds is not None:
        hex_pipe.ip_adapter_embeds = saved_ip_embeds

    observer.show_denoise_steps = True
    observer.on_log("info", "Pass 1 completed")

    # Reencode IP Adapter reference from pass 1 result
    if config.ip_adapter.enabled and config.ip_adapter.use_pass1_reference_for_pass2:
        ref_img = Image.fromarray(pass1)
        hex_pipe.encode_ip_reference(ref_img, config.diffusion.guidance_scale)
        observer.on_log("info", f"IP-Adapter reference re-encoded from pass 1 (scale={config.ip_adapter.scale})")

    observer.on_after_pass1(rgb_arr, mask_arr, pass1, wrapper, config.wrapper.output_size)

    # Pass 2 mask: full mask clipped to hex interior
    raw_W = wrapper.sq_right - wrapper.sq_left
    raw_H = wrapper.sq_bottom - wrapper.sq_top
    lat_scale = ((gen_W / raw_W) + (gen_H / raw_H)) / 2.0 if raw_W > 0 and raw_H > 0 else 1.0
    R_pixel = wrapper.R_cam * lat_scale

    _gy, _gx = np.mgrid[0:gen_H, 0:gen_W]
    hex_interior = _in_origin_hex(
        _gx.astype(np.float64) - gen_W / 2.0,
        _gy.astype(np.float64) - gen_H / 2.0,
        R_pixel,
    )
    pass2_mask = mask_arr.copy()
    pass2_mask[~hex_interior] = 0

    if pass2_mask.max() == 0:
        observer.on_log("info", "No mask inside hex — skipping pass 2")
        return pass1

    mask_f = mask_arr.astype(np.float32)[..., np.newaxis] / 255.0
    ctrl_composite = (
        rgb_arr.astype(np.float32) * (1.0 - mask_f)
        + pass1.astype(np.float32) * mask_f
    ).clip(0, 255).astype(np.uint8)

    # Pass 2: reinpaint full mask inside hex. rolling ON, ControlNet ON
    observer.on_log("info", "Pass 2: hex-periodic re-inpaint (rolling + ControlNet)")
    observer.on_before_pass2(pass1, pass2_mask)
    pass2 = hex_pipe.inpaint(
        source_image=pass1,
        mask_image=pass2_mask,
        prompt=dc.prompt,
        negative_prompt=dc.negative_prompt,
        gen_size=(gen_W, gen_H),
        wrapper=wrapper,
        control_image=ctrl_composite,
        use_latent_color_correction=config.postprocess.use_latent_color_correction,
        observer=observer,
        output_dir=output_path,
    )
    observer.on_log("info", "Pass 2 completed")

    return pass2
