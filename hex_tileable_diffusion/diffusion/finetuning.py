import gc
import math
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler  # type: ignore[import-not-found]
from PIL import Image, ImageEnhance

from hex_tileable_diffusion.config import FinetuneConfig
from hex_tileable_diffusion.observer.hexobserver import HexObserver


@dataclass
class FinetuneResult:
    losses: list[float]
    lora_active: bool


def _random_crop(image: Image.Image, crop_size: int, rng: random.Random) -> Image.Image:
    w, h = image.size
    side = rng.randint(crop_size, min(w, h))
    x0 = rng.randint(0, w - side) if w > side else 0
    y0 = rng.randint(0, h - side) if h > side else 0
    return image.crop((x0, y0, x0 + side, y0 + side)).resize(
        (crop_size, crop_size), Image.Resampling.LANCZOS,
    )


def _random_mask(
    H: int, W: int,
    ratio_range: tuple[float, float],
    rng: random.Random,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros(1, 1, H, W, device=device)
    ratio = rng.uniform(*ratio_range)
    kind = rng.choice(["rect", "cross", "border", "strips"])
    if kind == "rect":
        mh = max(16, min(int(H * math.sqrt(ratio)), H))
        mw = max(16, min(int(W * math.sqrt(ratio)), W))
        y0 = rng.randint(0, H - mh)
        x0 = rng.randint(0, W - mw)
        mask[0, 0, y0:y0 + mh, x0:x0 + mw] = 1.0
    elif kind == "cross":
        aw = max(8, int(W * ratio * 0.5))
        cy, cx = H // 2, W // 2
        mask[0, 0, max(0, cy - aw // 2):min(H, cy + aw // 2), :] = 1.0
        mask[0, 0, :, max(0, cx - aw // 2):min(W, cx + aw // 2)] = 1.0
    elif kind == "border":
        bd = max(4, int(H * ratio * 0.25))
        mask[0, 0, :bd, :] = 1.0
        mask[0, 0, -bd:, :] = 1.0
        mask[0, 0, :, :bd] = 1.0
        mask[0, 0, :, -bd:] = 1.0
    else:  # strips
        for _ in range(rng.randint(1, 4)):
            if rng.random() > 0.5:
                sh = max(8, int(H * ratio / 2))
                y0 = rng.randint(0, max(1, H - sh))
                mask[0, 0, y0:y0 + sh, :] = 1.0
            else:
                sw = max(8, int(W * ratio / 2))
                x0 = rng.randint(0, max(1, W - sw))
                mask[0, 0, :, x0:x0 + sw] = 1.0
    return mask


def _augment(image: Image.Image, rng: random.Random) -> Image.Image:
    k = rng.randint(0, 3)
    if k:
        image = image.rotate(k * 90, expand=False)
    if rng.random() > 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() > 0.5:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
    if rng.random() > 0.5:
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.9, 1.1))
    if rng.random() > 0.5:
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.9, 1.1))
    return image


def _encode_prompt(te: Any, tok: Any, text: str, device: torch.device) -> torch.Tensor:
    ids = tok(
        text, padding="max_length", max_length=tok.model_max_length,
        truncation=True, return_tensors="pt",
    ).input_ids.to(device)
    return te(ids)[0]


def finetune_pipeline_on_input(
    pipe: Any,
    input_image: Image.Image,
    prompt: str,
    config: FinetuneConfig,
    seed: int = 42,
    observer: HexObserver | None = None,
) -> FinetuneResult:
    device = pipe.device
    unet, vae, te, tok = pipe.unet, pipe.vae, pipe.text_encoder, pipe.tokenizer
    rng = random.Random(seed)
    crop_size = config.crop_size or 512

    # Reflect padding so random crops can span the tile's seam.
    pad = crop_size // 2
    padded = Image.fromarray(
        np.pad(np.array(input_image), ((pad, pad), (pad, pad), (0, 0)), mode="reflect"),
    )

    vae.requires_grad_(False)
    if te:
        te.requires_grad_(False)

    use_lora = config.use_lora
    if use_lora:
        try:
            from peft import LoraConfig
            unet.add_adapter(LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                init_lora_weights="gaussian",
                target_modules=["to_k", "to_q", "to_v", "to_out.0", "proj_in", "proj_out"],
            ))
        except ImportError:
            if observer:
                observer.on_log("warning", "peft not found → full finetune")
            use_lora = False

    if not use_lora:
        unet.requires_grad_(True)

    # AdamW needs fp32 master weights for stable updates.
    for p in unet.parameters():
        if p.requires_grad:
            p.data = p.data.float()

    trainable = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")  # type: ignore[attr-defined]
    result = FinetuneResult(losses=[], lora_active=use_lora)

    try:
        with torch.no_grad():
            prompt_emb = _encode_prompt(te, tok, prompt, device)
            empty_emb = _encode_prompt(te, tok, "", device)

        scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        in_channels = unet.config.in_channels
        unet.train()
        optimizer.zero_grad()
        running = 0.0

        for step in range(config.steps):
            src = padded if rng.random() > 0.3 else input_image
            crop = _random_crop(src, crop_size, rng)
            if config.use_augmentation:
                crop = _augment(crop, rng)

            image_t = (
                torch.from_numpy(np.array(crop)).float()
                .permute(2, 0, 1).unsqueeze(0) / 255.0 * 2.0 - 1.0
            ).to(device=device, dtype=torch.float16)

            with torch.no_grad():
                latents = vae.encode(image_t).latent_dist.sample().float() * vae.config.scaling_factor

            mask_px = _random_mask(crop_size, crop_size, config.mask_ratio_range, rng, device)
            mask_lat = F.interpolate(mask_px, size=latents.shape[2:], mode="nearest")
            masked_image = image_t * (1.0 - mask_px.to(image_t.dtype))

            with torch.no_grad():
                masked_latents = vae.encode(masked_image).latent_dist.sample().float() * vae.config.scaling_factor

            noise = torch.randn(latents.shape, device=device)
            timestep = torch.randint(0, scheduler.config.num_train_timesteps, (1,), device=device).long()
            noisy = scheduler.add_noise(latents, noise, timestep)

            unet_in = (
                torch.cat([noisy, mask_lat, masked_latents], dim=1).half()
                if in_channels == 9
                else noisy.half()
            )
            cond = (
                prompt_emb
                if (config.use_prompt_conditioning and rng.random() > config.prompt_dropout_prob)
                else empty_emb
            )

            with torch.amp.autocast("cuda", dtype=torch.float16):  # type: ignore[attr-defined]
                noise_pred = unet(unet_in, timestep, encoder_hidden_states=cond, return_dict=False)[0]

            loss = F.mse_loss(noise_pred.float(), noise)
            if in_channels == 9:
                # Up-weight loss inside the masked region (the part being inpainted).
                loss = 0.5 * loss + 0.5 * (((noise_pred.float() - noise) ** 2) * (1.0 + mask_lat)).mean()

            scaler.scale(loss / config.gradient_accumulation_steps).backward()  # type: ignore[no-untyped-call]
            if (step + 1) % config.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            result.losses.append(loss.item())
            running += loss.item()
            if observer and (step + 1) % config.log_interval == 0:
                observer.on_log(
                    "info",
                    f"Finetune step {step + 1}/{config.steps}, loss: {loss.item():.4f}, avg: {running / config.log_interval:.4f}",
                )
                running = 0.0

        unet.eval()
        if observer and result.losses:
            observer.on_log("info", f"Finetuning completed in {config.steps} steps. Final loss: {result.losses[-1]:.4f}")
        return result
    except Exception:
        cleanup_finetune(pipe, result, observer)
        raise
    finally:
        del optimizer, scaler, trainable
        torch.cuda.empty_cache()
        gc.collect()


def cleanup_finetune(
    pipe: Any,
    result: FinetuneResult,
    observer: HexObserver | None = None,
) -> None:
    unet = pipe.unet
    unet.eval()
    if result.lora_active:
        try:
            unet.delete_adapters("default")
            if observer:
                observer.on_log("info", "LoRA adapters deleted")
        except Exception as e:
            if observer:
                observer.on_log("warning", f"Failed to delete LoRA adapters: {e}")
    else:
        for p in unet.parameters():
            if p.dtype == torch.float32:
                p.data = p.data.half()
        if observer:
            observer.on_log(
                "warning",
                "Full finetune cleanup: base weights were modified — reload pipeline for a clean state",
            )
    unet.requires_grad_(False)
    torch.cuda.empty_cache()
    gc.collect()
