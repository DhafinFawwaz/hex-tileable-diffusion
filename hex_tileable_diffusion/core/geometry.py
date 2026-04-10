import numpy as np
from .constant import ONE_DIV_SQRT3, SQRT3, SQRT3_DIV_2
from typing import Union
import numpy.typing as npt

def _pixel_to_hex(px: np.ndarray, py: np.ndarray, R: float) -> tuple[np.ndarray, np.ndarray]:
    q = (ONE_DIV_SQRT3 * px - (1/3) * py) / R
    r = ((2/3) * py) / R
    return q, r


def _hex_to_pixel(q: np.ndarray, r: np.ndarray, R: float) -> tuple[np.ndarray, np.ndarray]:
    x = R * (SQRT3 * q + SQRT3_DIV_2 * r)
    y = R * 1.5 * r
    return x, y

def _cube_round(fq: np.ndarray, fr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fs = -fq - fr

    rq = np.round(fq).astype(float)
    rr = np.round(fr).astype(float)
    rs = np.round(fs).astype(float)

    dq = np.abs(rq - fq)
    dr = np.abs(rr - fr)
    ds = np.abs(rs - fs)

    mq = (dq > dr) & (dq > ds)
    mr = (~mq) & (dr > ds)

    rq[mq] = -rr[mq] - rs[mq]
    rr[mr] = -rq[mr] - rs[mr]
    return rq.astype(int), rr.astype(int)


def _hex_sdf(px: np.ndarray, py: np.ndarray, r_inscribed: float) -> np.ndarray:
    qx = np.abs(px)
    qy = np.abs(py)
    return np.maximum(qx, 0.5 * qx + SQRT3_DIV_2 * qy) - r_inscribed


def _box_sdf(px: np.ndarray, py: np.ndarray, half_x: float, half_y: float) -> np.ndarray:
    dx = np.abs(px) - half_x
    dy = np.abs(py) - half_y
    outside = np.sqrt(np.maximum(dx, 0.0) ** 2 + np.maximum(dy, 0.0) ** 2)
    inside = np.minimum(np.maximum(dx, dy), 0.0)
    return outside + inside


def _feather(dist: np.ndarray, threshold: float, feather_width: float) -> np.ndarray:
    if feather_width > 0:
        return np.clip(1.0 - (dist - threshold) / feather_width, 0.0, 1.0)
    return np.where(dist <= threshold, 1.0, 0.0)


def _compute_hex_grid(wx: np.ndarray, wy: np.ndarray, R_cam: float, r_inscribed: float, img_half_x: float, img_half_y: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fq, fr = _pixel_to_hex(wx, wy, R_cam)
    cq, cr = _cube_round(fq, fr)
    best_cx, best_cy = _hex_to_pixel(cq, cr, R_cam)

    # distances to hex center
    lx = wx - best_cx
    ly = wy - best_cy

    hs = _hex_sdf(lx, ly, r_inscribed)
    bs = _box_sdf(lx, ly, img_half_x, img_half_y)

    min_hex_sdf = hs
    min_content_sdf = np.maximum(hs, bs)

    return min_hex_sdf, min_content_sdf, best_cx, best_cy

def _sample_nearest(img_arr: npt.NDArray[np.uint8], u: np.ndarray, v: np.ndarray, valid_mask: npt.NDArray[np.bool_]) -> npt.NDArray[np.uint8]:
    H, W = img_arr.shape[:2]
    C = img_arr.shape[2]
    output = np.zeros(u.shape + (C,), dtype=np.uint8)

    src_x = np.clip((u * W).astype(np.int32), 0, W - 1)
    src_y = np.clip((v * H).astype(np.int32), 0, H - 1)
    output[valid_mask] = img_arr[src_y[valid_mask], src_x[valid_mask]]

    return output

def _tile_image_hexagonally(tile_arr: np.ndarray, out_w: int, out_h: int, R: float, offset_x: float = 0.0, offset_y: float = 0.0) -> np.ndarray:
    th, tw = tile_arr.shape[:2]
    tcx, tcy = tw / 2.0, th / 2.0

    gy, gx = np.mgrid[0:out_h, 0:out_w]
    px = gx.astype(np.float64) - offset_x
    py = gy.astype(np.float64) - offset_y

    fq, fr = _pixel_to_hex(px, py, R)
    rq, rr = _cube_round(fq, fr)
    cx, cy = _hex_to_pixel(rq, rr, R)

    src_x = np.clip(np.round(px - cx + tcx).astype(np.int32), 0, tw - 1)
    src_y = np.clip(np.round(py - cy + tcy).astype(np.int32), 0, th - 1)

    return tile_arr[src_y, src_x]

def _tile_image_square(tile_arr: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    th, tw = tile_arr.shape[:2]

    gy, gx = np.mgrid[0:out_h, 0:out_w]
    src_x = np.clip(np.round(gx % tw).astype(np.int32), 0, tw - 1)
    src_y = np.clip(np.round(gy % th).astype(np.int32), 0, th - 1)

    return tile_arr[src_y, src_x]