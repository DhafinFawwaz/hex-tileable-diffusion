import random
from typing import Any

import numpy as np
import torch
from PIL import Image

from hex_tileable_diffusion.core.geometry import _cube_round as cube_round
from hex_tileable_diffusion.core.hexwrapper import HexWrapper
from hex_tileable_diffusion.core.hexroller import hex_copy_fill_tensor, _random_hex_offset, roll_tensor_mode, unroll_tensor_mode
from hex_tileable_diffusion.diffusion.scheduling import interpolate_schedule
from hex_tileable_diffusion.observer.hexobserver import HexObserver
from diffusers import StableDiffusionInpaintPipeline
from hex_tileable_diffusion.types import RollMode


def run_rolling_inpaint(
    pipe: StableDiffusionInpaintPipeline,
    source_image: np.ndarray,
    mask_image: np.ndarray,
    prompt: str,
    negative_prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    strength: float,
    seed: int,
    gen_size: tuple[int, int],
    wrapper: HexWrapper,
    use_rolling_noise: bool = True,
    roll_mode: RollMode = "hex_copy",
    guidance_schedule: list[float] | None = None,
    controlnet: Any = None,
    control_image: np.ndarray | None = None,
    controlnet_conditioning_scale: float = 0.8,
    ip_adapter_image_embeds: list[torch.Tensor] | None = None,
    use_latent_color_correction: bool = False,
    vae_fp32: bool = False,
    observer: HexObserver = None,
    output_dir: str = ".",
) -> np.ndarray:
    S_W, S_H = gen_size
    dev = pipe.device
    nc = pipe.unet.config.in_channels
    do_cfg = guidance_scale > 1.0
    gen = torch.Generator(device="cuda").manual_seed(seed)
    rng = random.Random(seed)
    lat_W = S_W // 8
    lat_H = S_H // 8

    is_hex = roll_mode in ("hex", "hex_copy", "hex_copy_no_roll")
    needs_hex_fill = roll_mode in ("hex_copy", "hex_copy_no_roll")
    no_roll = roll_mode == "hex_copy_no_roll"

    R_cam = wrapper.R_cam
    raw_W = wrapper.sq_right - wrapper.sq_left
    raw_H = wrapper.sq_bottom - wrapper.sq_top
    lat_scale_x = S_W / raw_W if raw_W > 0 else 1.0
    lat_scale_y = S_H / raw_H if raw_H > 0 else 1.0
    lat_scale = (lat_scale_x + lat_scale_y) / 2.0
    R_lat = float(R_cam) * lat_scale / 8.0
    R_pixel = R_lat * 8.0

    # Convert to PIL to match pipe().__call__ preprocessing behavior
    # (diffusers 0.37+ handles numpy arrays differently from PIL in VaeImageProcessor)
    _src_pil = Image.fromarray(source_image) if isinstance(source_image, np.ndarray) else source_image
    _msk_pil = Image.fromarray(mask_image, mode="L") if isinstance(mask_image, np.ndarray) else mask_image

    # Encode text prompts
    with torch.no_grad():
        pe, ne = pipe.encode_prompt(
            prompt=prompt, device=dev, num_images_per_prompt=1,
            do_classifier_free_guidance=do_cfg,
            negative_prompt=negative_prompt,
        )
        pe_comb = torch.cat([ne, pe]) if do_cfg else pe

    # Scheduler timesteps
    pipe.scheduler.set_timesteps(num_inference_steps, device=dev)
    timesteps, actual_steps = pipe.get_timesteps(
        num_inference_steps=num_inference_steps,
        strength=strength, device=dev,
    )

    # Preprocess source image
    init_img = pipe.image_processor.preprocess(_src_pil, height=S_H, width=S_W).to(dtype=torch.float32)

    # Encode source image into latent space
    with torch.no_grad():
        n_lat = pipe.vae.config.latent_channels
        ret_il = nc == 4
        out = pipe.prepare_latents(
            1, n_lat, S_H, S_W, pe_comb.dtype, dev, gen, None,
            image=init_img, timestep=timesteps[:1],
            is_strength_max=(strength == 1.0),
            return_noise=True, return_image_latents=ret_il,
        )
        if ret_il:
            latents, noise, image_latents = out
        else:
            latents, noise = out
            image_latents = pipe.vae.encode(init_img.to(device=dev, dtype=torch.float16)).latent_dist.mean * pipe.vae.config.scaling_factor
            noise = torch.randn(latents.shape, device=latents.device, dtype=latents.dtype, generator=gen)

    # Prepare inpainting mask in latent space
    with torch.no_grad():
        mask_cond = pipe.mask_processor.preprocess(_msk_pil, height=S_H, width=S_W)
        masked_img = init_img * (mask_cond < 0.5)
        prep_mask, masked_lat = pipe.prepare_mask_latents(mask_cond, masked_img, 1, S_H, S_W, pe_comb.dtype, dev, gen, do_cfg)
        blend_mask = prep_mask[:1] if do_cfg else prep_mask

    # Prepare ControlNet input
    ctrl_img_tensor: torch.Tensor | None = None
    if controlnet is not None and control_image is not None:
        _ctrl_pil = Image.fromarray(control_image) if isinstance(control_image, np.ndarray) else control_image
        ctrl_img_tensor = pipe.image_processor.preprocess(_ctrl_pil, height=S_H, width=S_W).to(device=dev, dtype=torch.float16)
        ctrl_img_tensor = (ctrl_img_tensor + 1.0) / 2.0

    # Make all tensors hex-periodic
    if use_rolling_noise and needs_hex_fill:
        image_latents = hex_copy_fill_tensor(image_latents, R_lat)
        _ml_single = masked_lat[:1] if masked_lat.shape[0] > 1 else masked_lat
        _ml_filled = hex_copy_fill_tensor(_ml_single, R_lat)
        masked_lat = torch.cat([_ml_filled, _ml_filled], dim=0) if masked_lat.shape[0] > 1 else _ml_filled
        noise = hex_copy_fill_tensor(noise, R_lat)
        latents = hex_copy_fill_tensor(latents, R_lat)
        _pm = prep_mask[:1] if prep_mask.shape[0] > 1 else prep_mask
        _pm = hex_copy_fill_tensor(_pm, R_lat)
        prep_mask = torch.cat([_pm, _pm], dim=0) if prep_mask.shape[0] > 1 else _pm
        blend_mask = hex_copy_fill_tensor(blend_mask, R_lat)
        if ctrl_img_tensor is not None:
            ctrl_img_tensor = hex_copy_fill_tensor(ctrl_img_tensor, R_pixel)

    def _ts(name, t):
        print(f"  [DEBUG] {name}: shape={list(t.shape)} mean={t.float().mean().item():.6f} std={t.float().std().item():.6f} min={t.float().min().item():.6f} max={t.float().max().item():.6f}")
    _ts("latents", latents)
    _ts("image_latents", image_latents)
    _ts("noise", noise)
    _ts("prep_mask", prep_mask)
    _ts("blend_mask", blend_mask)
    _ts("masked_lat", masked_lat)
    print(f"  [DEBUG] R_lat={R_lat:.6f} R_pixel={R_pixel:.6f} actual_steps={actual_steps}")

    num_dd_steps = len(timesteps)
    dd_thresholds = 1.0 - torch.arange(1, num_dd_steps + 1, dtype=blend_mask.dtype, device=dev) / num_dd_steps  # [(N-1)/N ... 0]
    dd_step_masks = (blend_mask > dd_thresholds.view(-1, 1, 1, 1)).to(dtype=blend_mask.dtype)  # per-step binary activation

    # Denoising loop
    observer.on_log("debug", f"Starting denoising loop ({actual_steps} steps)")
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            cfg = guidance_scale
            if guidance_schedule and len(guidance_schedule) > 1:
                cfg = interpolate_schedule(guidance_schedule, i, actual_steps)

            # Differential Diffusion. inject noised original for not yet active pixels
            dd_mask_i = dd_step_masks[i]  # 1=denoising, 0=frozen
            init_at_t = pipe.scheduler.add_noise(image_latents, noise, torch.tensor([t]))  # original at current noise level
            if use_rolling_noise and needs_hex_fill:
                init_at_t = hex_copy_fill_tensor(init_at_t, R_lat)
            latents = dd_mask_i * latents + (1 - dd_mask_i) * init_at_t  # staggered activation
            if use_rolling_noise and needs_hex_fill:
                latents = hex_copy_fill_tensor(latents, R_lat)

            # Roll tensors
            if use_rolling_noise:
                if is_hex:
                    dx_lat, dy_lat = _random_hex_offset(rng, R_lat)
                else:
                    dx_lat = rng.randint(-lat_W // 2, lat_W // 2)
                    dy_lat = rng.randint(-lat_H // 2, lat_H // 2)
                if no_roll:
                    dx_lat, dy_lat = 0, 0
                dx_px = dx_lat * 8
                dy_px = dy_lat * 8

                if no_roll:
                    lat_r, mask_r, mimg_r = latents, prep_mask, masked_lat
                else:
                    lat_r = roll_tensor_mode(latents, dx_lat, dy_lat, R_lat, roll_mode, original=latents)
                    mask_r = roll_tensor_mode(prep_mask, dx_lat, dy_lat, R_lat, roll_mode, original=prep_mask)
                    mimg_r = roll_tensor_mode(masked_lat, dx_lat, dy_lat, R_lat, roll_mode, original=masked_lat)

                if no_roll:
                    ctrl_r = ctrl_img_tensor
                elif ctrl_img_tensor is not None:
                    ctrl_r = roll_tensor_mode(ctrl_img_tensor, dx_lat * 8, dy_lat * 8, R_pixel, roll_mode, original=ctrl_img_tensor)
                else:
                    ctrl_r = None
            else:
                dx_lat, dy_lat = 0, 0
                dx_px, dy_px = 0, 0
                lat_r = latents
                mask_r = prep_mask
                mimg_r = masked_lat
                ctrl_r = ctrl_img_tensor

            # Prepare UNet input
            lin_base = torch.cat([lat_r] * 2) if do_cfg else lat_r
            lin_base = pipe.scheduler.scale_model_input(lin_base, t)

            # ControlNet forward pass
            unet_extra: dict[str, Any] = {}
            if controlnet is not None and ctrl_r is not None:
                cn_dtype = controlnet.dtype
                ctrl_input = (torch.cat([ctrl_r] * 2) if do_cfg else ctrl_r).to(dtype=cn_dtype)
                down_samples, mid_sample = controlnet(
                    lin_base.to(dtype=cn_dtype), t,
                    encoder_hidden_states=pe_comb.to(dtype=cn_dtype),
                    controlnet_cond=ctrl_input,
                    conditioning_scale=controlnet_conditioning_scale,
                    return_dict=False,
                )
                unet_extra["down_block_additional_residuals"] = [s.to(dtype=pe_comb.dtype) for s in down_samples]
                unet_extra["mid_block_additional_residual"] = mid_sample.to(dtype=pe_comb.dtype)

            # IP Adapter embeddings
            if ip_adapter_image_embeds is not None:
                unet_dtype = pe_comb.dtype
                if isinstance(ip_adapter_image_embeds, list):
                    casted = [e.to(dtype=unet_dtype) for e in ip_adapter_image_embeds]
                else:
                    casted = ip_adapter_image_embeds.to(dtype=unet_dtype)
                unet_extra["added_cond_kwargs"] = {"image_embeds": casted}

            # 9-channel inpaint: concatenate mask + masked image
            if nc == 9:
                lin = torch.cat([lin_base, mask_r, mimg_r], dim=1)
            else:
                lin = lin_base

            # UNet forward pass
            pred = pipe.unet(lin, t, encoder_hidden_states=pe_comb, **unet_extra, return_dict=False)[0]

            # Classifier free guidance
            if do_cfg:
                pu, pc = pred.chunk(2)
                pred = pu + cfg * (pc - pu)

            # Scheduler step
            lat_r = pipe.scheduler.step(pred, t, lat_r, return_dict=False)[0]

            # Reenforce hex periodicity
            if use_rolling_noise and needs_hex_fill:
                lat_r = hex_copy_fill_tensor(lat_r, R_lat)
            observer.on_log("debug", f"Step {i + 1}/{len(timesteps)}| ({dx_lat}, {dy_lat}) latents, ({dx_px}, {dy_px}) pixels")
            observer.on_denoise_step(i, len(timesteps), pipe.vae, pipe.image_processor, lat_r, mask_r, mimg_r, ctrl_r)

            # Unroll
            if use_rolling_noise:
                latents = unroll_tensor_mode(lat_r, dx_lat, dy_lat, R_lat, roll_mode)
            else:
                latents = lat_r


    # Differential Diffusion: final blend to lock mask=0 to original
    latents = blend_mask * latents + (1 - blend_mask) * image_latents  # hard clamp at boundary
    if use_rolling_noise and needs_hex_fill:
        latents = hex_copy_fill_tensor(latents, R_lat)

    # Latent-space color correction via AdaIN
    if use_latent_color_correction:
        with torch.no_grad():
            eps = 1e-6
            tgt_mean = image_latents.mean(dim=[2, 3], keepdim=True)
            tgt_std = image_latents.std(dim=[2, 3], keepdim=True) + eps
            src_mean = latents.mean(dim=[2, 3], keepdim=True)
            src_std = latents.std(dim=[2, 3], keepdim=True) + eps
            corrected = (latents - src_mean) / src_std * tgt_std + tgt_mean
            latents = (1 - blend_mask) * latents + blend_mask * corrected
            if use_rolling_noise and needs_hex_fill:
                latents = hex_copy_fill_tensor(latents, R_lat)

    observer.on_log("debug", "Denoising loop completed, decoding latents to image...")
    # Decode latents to pixels
    with torch.no_grad():
        decode_latents = latents / pipe.vae.config.scaling_factor
        if vae_fp32:
            pipe.vae.to(dtype=torch.float32)
            decoded = pipe.vae.decode(decode_latents.float(), return_dict=False)[0]
            pipe.vae.to(dtype=torch.float16)
        else:
            decoded = pipe.vae.decode(decode_latents, return_dict=False)[0]
        if use_rolling_noise and needs_hex_fill:
            decoded = hex_copy_fill_tensor(decoded, R_pixel)

    result_pil = pipe.image_processor.postprocess(decoded, output_type="pil")[0]
    return np.array(result_pil)
