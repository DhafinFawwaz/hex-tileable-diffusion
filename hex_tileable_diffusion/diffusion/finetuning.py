import gc
import math
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance

from hex_tileable_diffusion.config import FinetuneConfig
from hex_tileable_diffusion.observer.hexobserver import HexObserver


@dataclass
class FinetuneResult:
    losses: list[float]
    lora_active: bool


def _generate_random_crop(image: Image.Image, crop_size: int, rng: random.Random) -> Image.Image:
    w, h = image.size
    ms = min(w, h)
    cd = rng.randint(crop_size, ms)
    x0 = rng.randint(0, w - cd) if w > cd else 0
    y0 = rng.randint(0, h - cd) if h > cd else 0
    return image.crop((x0, y0, x0 + cd, y0 + cd)).resize(
        (crop_size, crop_size), Image.Resampling.LANCZOS,
    )


def _generate_random_mask_tensor(
    batch: int,
    H: int, W: int,
    mask_ratio_range: tuple[float, float],
    rng: random.Random,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    masks = torch.zeros(batch, 1, H, W, device=device, dtype=dtype)
    for b in range(batch):
        ratio = rng.uniform(*mask_ratio_range)
        mt = rng.choice(["rect", "cross", "border", "strips"])
        if mt == "rect":
            mh = max(16, min(int(H * math.sqrt(ratio)), H))
            mw = max(16, min(int(W * math.sqrt(ratio)), W))
            y0 = rng.randint(0, H - mh + 1)
            x0 = rng.randint(0, W - mw + 1)
            masks[b, 0, y0 : y0 + mh, x0 : x0 + mw] = 1.0
        elif mt == "cross":
            aw = max(8, int(W * ratio * 0.5))
            cy, cx = H // 2, W // 2
            masks[b, 0, max(0, cy - aw // 2) : min(H, cy + aw // 2), :] = 1.0
            masks[b, 0, :, max(0, cx - aw // 2) : min(W, cx + aw // 2)] = 1.0
        elif mt == "border":
            bd = max(4, int(H * ratio * 0.25))
            masks[b, 0, :bd, :] = 1.0
            masks[b, 0, -bd:, :] = 1.0
            masks[b, 0, :, :bd] = 1.0
            masks[b, 0, :, -bd:] = 1.0
        elif mt == "strips":
            for _ in range(rng.randint(1, 4)):
                if rng.random() > 0.5:
                    sh = max(8, int(H * ratio / 2))
                    y0 = rng.randint(0, max(1, H - sh))
                    masks[b, 0, y0 : y0 + sh, :] = 1.0
                else:
                    sw = max(8, int(W * ratio / 2))
                    x0 = rng.randint(0, max(1, W - sw))
                    masks[b, 0, :, x0 : x0 + sw] = 1.0
    return masks


def _apply_training_augmentation(image: Image.Image, rng: random.Random) -> Image.Image:
    c = image
    k = rng.randint(0, 3)
    if k > 0:
        c = c.rotate(k * 90, expand=False)
    if rng.random() > 0.5:
        c = c.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() > 0.5:
        c = c.transpose(Image.FLIP_TOP_BOTTOM)
    if rng.random() > 0.5:
        c = ImageEnhance.Brightness(c).enhance(rng.uniform(0.9, 1.1))
    if rng.random() > 0.5:
        c = ImageEnhance.Contrast(c).enhance(rng.uniform(0.9, 1.1))
    return c


def finetune_pipeline_on_input(
    pipe: Any,
    input_image: Image.Image,
    prompt: str,
    config: FinetuneConfig,
    seed: int = 42,
    observer: HexObserver | None = None,
) -> FinetuneResult:
    dev = pipe.device
    unet = pipe.unet
    vae = pipe.vae
    te = pipe.text_encoder
    tok = pipe.tokenizer
    rng = random.Random(seed)
    crop_size = config.crop_size if config.crop_size else 512

    inp = np.array(input_image)
    pad = crop_size // 2
    padded = Image.fromarray(
        np.pad(inp, ((pad, pad), (pad, pad), (0, 0)), mode="reflect"),
    )

    vae.requires_grad_(False)
    if te:
        te.requires_grad_(False)

    la = False
    losses: list[float] = []
    success = False
    try:
        use_lora = config.use_lora
        if use_lora:
            try:
                from peft import LoraConfig

                unet.add_adapter(
                    LoraConfig(
                        r=config.lora_rank,
                        lora_alpha=config.lora_alpha,
                        init_lora_weights="gaussian",
                        target_modules=[
                            "to_k", "to_q", "to_v", "to_out.0",
                            "proj_in", "proj_out",
                        ],
                    ),
                )
                la = True
            except ImportError:
                if observer is not None:
                    observer.on_log("warning", "peft not found → full finetune")
                use_lora = False

        if not use_lora:
            unet.requires_grad_(True)

        for p in unet.parameters():
            if p.requires_grad:
                p.data = p.data.float()

        tp = [p for p in unet.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(tp, lr=config.lr, weight_decay=1e-4)
        gs = torch.amp.GradScaler("cuda")  # type: ignore[attr-defined]

        with torch.no_grad():
            pe = te(
                tok(
                    prompt, padding="max_length",
                    max_length=tok.model_max_length,
                    truncation=True, return_tensors="pt",
                ).input_ids.to(dev),
            )[0]
            ue = te(
                tok(
                    "", padding="max_length",
                    max_length=tok.model_max_length,
                    truncation=True, return_tensors="pt",
                ).input_ids.to(dev),
            )[0]

        from diffusers import DDPMScheduler  # type: ignore[import-not-found]

        ts = DDPMScheduler.from_config(pipe.scheduler.config)
        unet.train()
        rl = 0.0
        nc = unet.config.in_channels
        opt.zero_grad()
        num_steps = config.steps

        for step in range(num_steps):
            src = padded if rng.random() > 0.3 else input_image
            crop = _generate_random_crop(src, crop_size, rng)
            if config.use_augmentation:
                crop = _apply_training_augmentation(crop, rng)

            cn = (
                torch.from_numpy(np.array(crop)).float()
                .permute(2, 0, 1).unsqueeze(0) / 255.0 * 2.0 - 1.0
            ).to(device=dev, dtype=torch.float16)

            with torch.no_grad():
                lat = (
                    vae.encode(cn).latent_dist.sample().float()
                    * vae.config.scaling_factor
                )
            lh, lw = lat.shape[2], lat.shape[3]
            mp = _generate_random_mask_tensor(
                1, crop_size, crop_size, config.mask_ratio_range, rng, dev,
            )
            ml = F.interpolate(mp, size=(lh, lw), mode="nearest").float()
            mi = cn * (1.0 - mp.to(dtype=cn.dtype, device=dev))

            with torch.no_grad():
                mls = (
                    vae.encode(mi).latent_dist.sample().float()
                    * vae.config.scaling_factor
                )

            noise = torch.randn(lat.shape, device=dev, dtype=torch.float32)
            timestep = torch.randint(
                0, ts.config.num_train_timesteps, (1,), device=dev,
            ).long()
            nl = ts.add_noise(lat, noise, timestep)

            ui = (
                torch.cat([nl, ml, mls], dim=1).half()
                if nc == 9
                else nl.half()
            )
            ehs = (
                pe
                if (config.use_prompt_conditioning and rng.random() > config.prompt_dropout_prob)
                else ue
            )

            with torch.amp.autocast("cuda", dtype=torch.float16):  # type: ignore[attr-defined]
                np_ = unet(
                    ui, timestep, encoder_hidden_states=ehs, return_dict=False,
                )[0]

            loss = F.mse_loss(np_.float(), noise)
            if nc == 9:
                loss = 0.5 * loss + 0.5 * (
                    ((np_.float() - noise) ** 2) * (1.0 + ml)
                ).mean()

            gs.scale(loss / config.gradient_accumulation_steps).backward()  # type: ignore[no-untyped-call]
            if (step + 1) % config.gradient_accumulation_steps == 0:
                gs.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(tp, 1.0)
                gs.step(opt)
                gs.update()
                opt.zero_grad()

            losses.append(loss.item())
            rl += loss.item()
            if observer is not None and (step + 1) % config.log_interval == 0:
                observer.on_log("info", f"Finetune step {step + 1}/{num_steps}, loss: {loss.item():.4f}, avg loss: {rl / config.log_interval:.4f}")
                rl = 0.0

        if num_steps % config.gradient_accumulation_steps != 0:
            gs.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(tp, 1.0)
            gs.step(opt)
            gs.update()

        unet.eval()
        del opt, gs, tp
        torch.cuda.empty_cache()
        gc.collect()

        if observer is not None and losses:
            observer.on_log("info", f"Finetuning completed in {num_steps} steps. Final loss: {losses[-1]:.4f}")

        success = True
        return FinetuneResult(losses=losses, lora_active=la)
    finally:
        if not success:
            try:
                cleanup_finetune(pipe, FinetuneResult(losses=losses, lora_active=la), observer)
            except Exception as cleanup_err:
                if observer is not None:
                    observer.on_log("warning", f"finetune rollback failed: {cleanup_err}")


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
            if observer is not None:
                observer.on_log("info", "LoRA adapters deleted")
        except Exception as e:
            if observer is not None:
                observer.on_log("warning", f"Failed to delete LoRA adapters: {e}")
    else:
        for p in unet.parameters():
            if p.dtype == torch.float32:
                p.data = p.data.half()
        if observer is not None:
            observer.on_log(
                "warning",
                "Full finetune cleanup: base weights were modified in place — reload pipeline for a clean state",
            )
    unet.requires_grad_(False)
    torch.cuda.empty_cache()
    gc.collect()
