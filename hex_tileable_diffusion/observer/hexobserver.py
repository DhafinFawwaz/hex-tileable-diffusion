from time import time
from typing import Any

import numpy as np
import torch
from PIL import Image
from IPython.display import display

from hex_tileable_diffusion.types import LogLevel
from hex_tileable_diffusion.core.constant import SQRT3
from hex_tileable_diffusion.core.geometry import _hex_sdf, _tile_image_hexagonally, _tile_image_square
from hex_tileable_diffusion.core.hexwrapper import HexWrapper
from hex_tileable_diffusion.util.decode import decode_latents_to_image


def format_print(v: Any) -> str:
    if isinstance(v, dict): return "\n".join(f"{k}: {v}" for k, v in v.items())
    elif hasattr(v, "__dict__"): return format_print(v.__dict__)
    else: return str(v)

def format_time(t: float) -> str:
    """Format to MM:SS.SS"""
    minutes = int(t // 60)
    seconds = int(t % 60)
    milliseconds = int((t % 1) * 100)
    return f"{minutes:02d}:{seconds:02d}.{milliseconds:02d}"


def _to_rgba(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        rgba = np.zeros((*img.shape, 4), dtype=np.uint8)
        rgba[..., 0] = rgba[..., 1] = rgba[..., 2] = img
        rgba[..., 3] = 255
        return rgba
    if img.shape[2] == 4:
        return img
    rgba = np.zeros((*img.shape[:2], 4), dtype=np.uint8)
    rgba[..., :3] = img[..., :3]
    rgba[..., 3] = 255
    return rgba


def _concat_horizontal(images: list[np.ndarray], padding: int = 4) -> Image.Image:
    if not images:
        return Image.new("RGBA", (1, 1))
    processed = [_to_rgba(img) for img in images]
    max_h = max(img.shape[0] for img in processed)
    resized = []
    for img in processed:
        if img.shape[0] != max_h:
            scale = max_h / img.shape[0]
            new_w = int(img.shape[1] * scale)
            img = np.array(Image.fromarray(img, mode="RGBA").resize((new_w, max_h), Image.Resampling.LANCZOS))
        resized.append(img)
    total_w = sum(img.shape[1] for img in resized) + padding * (len(resized) - 1)
    canvas = np.zeros((max_h, total_w, 4), dtype=np.uint8)
    x = 0
    for img in resized:
        h, w = img.shape[:2]
        canvas[:h, x:x+w] = img
        x += w + padding
    return Image.fromarray(canvas, mode="RGBA")


def _draw_hex_contour(img_arr: np.ndarray, R: float, thickness: float = 2.0, color: tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
    h, w = img_arr.shape[:2]
    result = img_arr.copy()
    gy, gx = np.mgrid[0:h, 0:w]
    hx = gx.astype(np.float64) - w / 2.0
    hy = gy.astype(np.float64) - h / 2.0
    r_inscribed = (SQRT3 / 2.0) * R
    sdf = _hex_sdf(hx, hy, r_inscribed)
    contour_mask = np.abs(sdf) < thickness
    result[contour_mask] = color
    return result


class HexObserver():
    start_time: float = 0
    preview_count: int = 3
    show_denoise_steps: bool = True

    def __init__(self):
        self.start_time = 0
        self.preview_count = 3
        self.show_denoise_steps = True

    def on_start(self) -> None:
        self.start_time = time()
        self.on_log("info", "Hexagonal Seamless & Tileable Texture Diffusion Generation started")

    def on_log(self, level: LogLevel, message: str, values: dict | Any | None = None) -> None:
        t = time() - self.start_time if self.start_time else 0
        print(f"[{level.upper()}] [{format_time(t)}] {message}")
        if values is not None: print(format_print(values))

    def visualize_wrapped_finished(self, wrapper: HexWrapper, rgb_arr: np.ndarray, mask_arr: np.ndarray, hex_outline_thickness: float) -> list[np.ndarray]:
        debug_img = wrapper.debug_wrap(rgb_arr, mask_arr, hex_outline_thickness)
        return [debug_img[..., :3], rgb_arr, np.stack([mask_arr] * 3, axis=-1)]

    def on_wrapped_finished(self, wrapper: HexWrapper, rgb_arr: np.ndarray, mask_arr: np.ndarray, hex_outline_thickness: float) -> None:
        self.on_log("info", "Wrapped")
        print("Canvas Debugger | RGB (Hex Cam View) | Mask")
        display(_concat_horizontal(self.visualize_wrapped_finished(wrapper, rgb_arr, mask_arr, hex_outline_thickness)))

    def visualize_after_pass1(self, rgb_arr: np.ndarray, mask_arr: np.ndarray, pass1_result: np.ndarray, wrapper: HexWrapper, output_size: int) -> list[np.ndarray]:
        tiled_result, R = wrapper.unwrap(pass1_result, output_size=output_size)
        tiled_arr = _tile_image_hexagonally(tiled_result, output_size * 2, output_size * 2, R)
        return [rgb_arr, np.stack([mask_arr] * 3, axis=-1), pass1_result, tiled_arr.astype(np.uint8)]

    def on_after_pass1(self, rgb_arr: np.ndarray, mask_arr: np.ndarray, pass1_result: np.ndarray, wrapper: HexWrapper, output_size: int) -> None:
        self.on_log("info", "After Pass 1")
        print("Before | Mask | After | Tiled")
        display(_concat_horizontal(self.visualize_after_pass1(rgb_arr, mask_arr, pass1_result, wrapper, output_size)))

    def visualize_before_pass2(self, pass1_result: np.ndarray, pass2_mask: np.ndarray) -> list[np.ndarray]:
        return [pass1_result, np.stack([pass2_mask] * 3, axis=-1)]

    def on_before_pass2(self, pass1_result: np.ndarray, pass2_mask: np.ndarray) -> None:
        self.on_log("info", "Before Pass 2")
        print("Pass 1 Result | Pass 2 Mask")
        display(_concat_horizontal(self.visualize_before_pass2(pass1_result, pass2_mask)))

    def visualize_denoise_step(self, step: int, total_steps: int, vae: Any, image_processor: Any, lat_r: torch.Tensor, mask_r: torch.Tensor, mimg_r: torch.Tensor, ctrl_r: torch.Tensor | None) -> list[np.ndarray]:
        images = []

        mask_np = mask_r[0, 0].cpu().float().numpy()
        mask_vis = (mask_np * 255).astype(np.uint8)
        mask_vis = np.array(Image.fromarray(mask_vis).resize(
            (mask_vis.shape[1] * 8, mask_vis.shape[0] * 8), Image.Resampling.NEAREST,
        ))
        images.append(mask_vis)

        if ctrl_r is not None:
            ctx_np = ctrl_r[0].cpu().float().permute(1, 2, 0).numpy()
            ctx_vis = (ctx_np.clip(0, 1) * 255).astype(np.uint8)
        else:
            ctx_vis = decode_latents_to_image(vae, image_processor, mimg_r[:1])
        images.append(ctx_vis)

        images.append(decode_latents_to_image(vae, image_processor, lat_r))

        return images

    def on_denoise_step(self, step: int, total_steps: int, vae: Any, image_processor: Any, lat_r: torch.Tensor, mask_r: torch.Tensor, mimg_r: torch.Tensor, ctrl_r: torch.Tensor | None) -> None:
        if not self.show_denoise_steps:
            return
        if self.preview_count <= 0:
            return
        interval = max(1, total_steps // self.preview_count)
        is_last = (step == total_steps - 1)
        if step % interval != 0 and not is_last:
            return
        self.on_log("info", f"Denoise step {step + 1}/{total_steps} | Previewing...")
        print(f"Step {step + 1}/{total_steps}: Rolled Mask | Rolled Ctx | Denoised")
        display(_concat_horizontal(self.visualize_denoise_step(step, total_steps, vae, image_processor, lat_r, mask_r, mimg_r, ctrl_r)))

    def visualize_after_inpaint(self, rgb_arr: np.ndarray, mask_arr: np.ndarray, result: np.ndarray) -> list[np.ndarray]:
        overlay = rgb_arr[..., :3].copy().astype(np.float32)
        mask_f = mask_arr.astype(np.float32) / 255.0
        if mask_f.ndim == 2:
            mask_f = mask_f[..., np.newaxis]
        red = np.array([255, 0, 0], dtype=np.float32)
        overlay = overlay * (1 - mask_f * 0.5) + red * mask_f * 0.5
        overlay = overlay.clip(0, 255).astype(np.uint8)
        return [rgb_arr[..., :3], np.stack([mask_arr] * 3, axis=-1), overlay, result[..., :3]]

    def on_after_inpaint(self, rgb_arr: np.ndarray, mask_arr: np.ndarray, result: np.ndarray) -> None:
        self.on_log("info", "After Inpaint")
        print("Source | Mask | Overlay | Result")
        display(_concat_horizontal(self.visualize_after_inpaint(rgb_arr, mask_arr, result)))

    def visualize_after_unwrap(self, wrapper: HexWrapper, inpainted_rgb: np.ndarray, result: np.ndarray, R_final: float, output_size: int) -> list[np.ndarray]:
        gen_W, gen_H = wrapper.gen_W, wrapper.gen_H
        raw_W = wrapper.sq_right - wrapper.sq_left
        raw_H = wrapper.sq_bottom - wrapper.sq_top
        lat_scale = ((gen_W / raw_W) + (gen_H / raw_H)) / 2.0 if raw_W > 0 and raw_H > 0 else 1.0
        R_pixel = wrapper.R_cam * lat_scale
        contour_img = _draw_hex_contour(inpainted_rgb.copy(), R_pixel, thickness=2.0)

        unwrapped_img = result[..., :3]

        h, w = result.shape[:2]
        gy, gx = np.mgrid[0:h, 0:w]
        hx = gx.astype(np.float64) - w / 2.0
        hy = gy.astype(np.float64) - h / 2.0
        r_inscribed = (SQRT3 / 2.0) * R_final
        hex_mask = _hex_sdf(hx, hy, r_inscribed) < 0
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[..., :3] = result[..., :3]
        rgba[..., 3] = np.where(hex_mask, 255, 0).astype(np.uint8)

        return [contour_img, unwrapped_img, rgba]

    def on_after_unwrap(self, wrapper: HexWrapper, inpainted_rgb: np.ndarray, result: np.ndarray, R_final: float, output_size: int) -> None:
        self.on_log("info", "After Unwrap")
        print("Inpainted (hex contour) | Unwrapped | Hex Cropped")
        display(_concat_horizontal(self.visualize_after_unwrap(wrapper, inpainted_rgb, result, R_final, output_size)))

    def visualize_after_postprocess(self, before_postprocess: np.ndarray, after_postprocess: np.ndarray) -> list[np.ndarray]:
        return [before_postprocess[..., :3], after_postprocess[..., :3]]

    def on_after_postprocess(self, before_postprocess: np.ndarray, after_postprocess: np.ndarray) -> None:
        self.on_log("info", "After Postprocess")
        print("Final | Postprocessed")
        display(_concat_horizontal(self.visualize_after_postprocess(before_postprocess, after_postprocess)))

    def visualize_finished(self, image_arr: np.ndarray, result: np.ndarray, R_final: float, output_size: int) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        tile_w, tile_h = output_size * 3, output_size * 3
        R_input = image_arr.shape[0] / 2.0

        hex_tiled_input = _tile_image_hexagonally(image_arr, tile_w, tile_h, R_input).astype(np.uint8)
        hex_tiled_output = _tile_image_hexagonally(result, tile_w, tile_h, R_final).astype(np.uint8)
        row1 = [hex_tiled_input, hex_tiled_output]

        sq_tiled_input = _tile_image_square(image_arr, tile_w, tile_h).astype(np.uint8)
        hex_tiled_output_lines = _draw_hex_contour(hex_tiled_output.copy(), R_final, thickness=1.5)
        row2 = [sq_tiled_input, hex_tiled_output_lines]

        row3 = [image_arr[..., :3], result[..., :3]]

        return row1, row2, row3

    def on_finished(self, image_arr: np.ndarray, result: np.ndarray, R_final: float, output_size: int) -> None:
        self.on_log("info", "Finished")
        row1, row2, row3 = self.visualize_finished(image_arr, result, R_final, output_size)
        print("Hex Tiled Input | Hex Tiled Output")
        display(_concat_horizontal(row1))
        print("Square Tiled Input | Hex Tiled Output (hex lines)")
        display(_concat_horizontal(row2))
        print("Input | Output")
        display(_concat_horizontal(row3))
