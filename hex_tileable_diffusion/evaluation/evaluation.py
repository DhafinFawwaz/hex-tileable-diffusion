import numpy as np
import torch
from PIL import Image

from hex_tileable_diffusion.core.geometry import _tile_image_hexagonally

from .metric import Metrics


def hex_tile_for_textile(img: np.ndarray, R: float) -> np.ndarray:
    right_r = R * np.sqrt(3) / 2
    w = int(round(right_r * 2))
    h = int(round(R * 3))
    return _tile_image_hexagonally(img, w, h, R)


def scale_match_to_reference(gen: np.ndarray, R: float, ref_w: int, ref_h: int) -> np.ndarray:
    gen_H, gen_W = gen.shape[:2]
    cy, cx = gen_H / 2.0, gen_W / 2.0
    y0 = max(0, int(round(cy - R)))
    y1 = min(gen_H, int(round(cy + R)))
    x0 = max(0, int(round(cx - R)))
    x1 = min(gen_W, int(round(cx + R)))
    crop = gen[y0:y1, x0:x1]
    resized = Image.fromarray(crop).resize((ref_w, ref_h), Image.Resampling.LANCZOS)
    return np.array(resized)


def _to_tensor(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0


class HexEvaluation:
    ssim_score: float
    lpips_score: float
    si_fid_score: float
    textile_score: float

    def __init__(
        self,
        reference_img_arr: np.ndarray,
        generated_img_arr: np.ndarray,
        hex_radius: float | None = None,
        metrics: Metrics | None = None,
    ):
        m = metrics if metrics is not None else Metrics()
        H, W = reference_img_arr.shape[:2]

        if hex_radius is not None:
            sim_gen = scale_match_to_reference(generated_img_arr, hex_radius, W, H)
            tex_gen = hex_tile_for_textile(generated_img_arr, hex_radius)
        else:
            sim_gen = generated_img_arr
            tex_gen = generated_img_arr

        ref_t = _to_tensor(reference_img_arr)
        gen_t = _to_tensor(sim_gen)

        self.ssim_score = m.ssim(ref_t, gen_t)
        self.lpips_score = m.lpips(ref_t, gen_t)
        self.si_fid_score = m.si_fid(ref_t, gen_t)
        self.textile_score = m.textile(tex_gen)


if __name__ == "__main__":
    ref_path = "demos/rock1_original_512.png"
    gen_path = "demos/rock1_output.png"

    ref_arr = np.array(Image.open(ref_path).convert("RGB"))
    gen_arr = np.array(Image.open(gen_path).convert("RGB"))

    R = gen_arr.shape[0] / 2.0

    print(f"reference : {ref_path}")
    print(f"generated : {gen_path}")
    print(f"hex_radius: {R}")
    print()

    print("raw (no scale-match, no hex-tile)")
    ev_raw = HexEvaluation(ref_arr, gen_arr)
    print(f"ssim   : {ev_raw.ssim_score:.6f}")
    print(f"lpips  : {ev_raw.lpips_score:.6f}")
    print(f"si_fid : {ev_raw.si_fid_score:.6f}")
    print(f"textile: {ev_raw.textile_score:.6f}")
    print()

    print(f"hex-aware (hex_radius={R})")
    ev_hex = HexEvaluation(ref_arr, gen_arr, hex_radius=R)
    print(f"ssim   : {ev_hex.ssim_score:.6f}")
    print(f"lpips  : {ev_hex.lpips_score:.6f}")
    print(f"si_fid : {ev_hex.si_fid_score:.6f}")
    print(f"textile: {ev_hex.textile_score:.6f}")
