import cv2
import numpy as np


def _rgb_to_lab(rgb_u8: np.ndarray) -> np.ndarray:
    rgb_f = rgb_u8.astype(np.float32) / 255.0
    return cv2.cvtColor(rgb_f, cv2.COLOR_RGB2LAB)


def _lab_to_rgb(lab_f32: np.ndarray) -> np.ndarray:
    rgb_f = cv2.cvtColor(lab_f32, cv2.COLOR_LAB2RGB)
    return (np.clip(rgb_f, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _match_stats(src: np.ndarray, ref: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    s_flat = src.reshape(-1, src.shape[-1])
    r_flat = ref.reshape(-1, ref.shape[-1])
    s_mean = s_flat.mean(axis=0)
    s_std = s_flat.std(axis=0)
    r_mean = r_flat.mean(axis=0)
    r_std = r_flat.std(axis=0)
    return (src - s_mean) / (s_std + eps) * r_std + r_mean


def _blur_lab(lab: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return lab
    return cv2.GaussianBlur(
        lab, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT,
    )


def low_freq_color_transfer(
    source_rgb: np.ndarray,
    reference_rgb: np.ndarray,
    blur_sigma: float = 32.0,
    strength: float = 1.0,
) -> np.ndarray:
    src_lab = _rgb_to_lab(source_rgb)
    ref_lab = _rgb_to_lab(reference_rgb)
    src_low = _blur_lab(src_lab, blur_sigma)
    ref_low = _blur_lab(ref_lab, blur_sigma)
    high = src_lab - src_low
    low_matched = _match_stats(src_low, ref_low)
    blended_low = (1.0 - strength) * src_low + strength * low_matched
    return _lab_to_rgb(blended_low + high)
