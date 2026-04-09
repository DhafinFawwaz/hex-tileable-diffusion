import math
import random

import numpy as np
import torch

from numpy.typing import NDArray

from hex_tileable_diffusion.core.geometry import _cube_round, _pixel_to_hex, _hex_to_pixel
from hex_tileable_diffusion.types import RollMode


def _in_origin_hex(px: NDArray[np.float64], py: NDArray[np.float64], R: float) -> NDArray[np.bool_]:
    q, r = _pixel_to_hex(px, py, R)
    cq, cr = _cube_round(q, r)
    return (cq == 0) & (cr == 0)


def _wrap_to_origin_hex(px: NDArray[np.float64], py: NDArray[np.float64], R: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    q, r = _pixel_to_hex(px, py, R)
    cq, cr = _cube_round(q.astype(np.float64), r.astype(np.float64))
    hcx, hcy = _hex_to_pixel(cq, cr, R)
    return px - hcx, py - hcy


def _resolve_collisions(
    src_x: NDArray[np.int32], src_y: NDArray[np.int32],
    float_x: NDArray[np.float64], float_y: NDArray[np.float64],
    hex_mask: NDArray[np.bool_], W: int,
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    
    # Snap sources outside hex_mask to nearest hex pixel
    outside = ~hex_mask[src_y, src_x]
    if outside.any():
        hex_ys, hex_xs = np.where(hex_mask)
        for i in np.where(outside)[0]:
            dists = (hex_xs - float_x[i])**2 + (hex_ys - float_y[i])**2
            nearest = np.argmin(dists)
            src_x[i] = hex_xs[nearest]
            src_y[i] = hex_ys[nearest]

    # Collision check
    targets = src_y.astype(np.int64) * W + src_x.astype(np.int64)
    if len(np.unique(targets)) == len(targets): return src_x, src_y

    # Lowest rounding error gets priority to claim a target
    errors = (float_x - src_x)**2 + (float_y - src_y)**2
    claimed = set()
    reassign = []
    for idx in np.argsort(errors):
        t = int(targets[idx])
        if t not in claimed: claimed.add(t)
        else: reassign.append(idx)

    # Reassign collided pixels to nearest unclaimed hex pixel
    hex_lin = set((np.where(hex_mask)[0].astype(np.int64) * W + np.where(hex_mask)[1].astype(np.int64)).tolist())
    avail = np.array(sorted(hex_lin - claimed))
    avail_x = (avail % W).astype(np.float64)
    avail_y = (avail // W).astype(np.float64)

    for i in reassign:
        dists = (avail_x - float_x[i])**2 + (avail_y - float_y[i])**2
        best = np.argmin(dists)
        src_x[i] = int(avail[best] % W)
        src_y[i] = int(avail[best] // W)
        keep = np.ones(len(avail), dtype=bool)
        keep[best] = False
        avail, avail_x, avail_y = avail[keep], avail_x[keep], avail_y[keep]

    return src_x, src_y


def _hex_roll_remap(H: int, W: int, dx: int, dy: int, R: float) -> tuple[NDArray[np.float64], NDArray[np.int32], NDArray[np.bool_]]:
    cx, cy = W / 2.0, H / 2.0

    gy, gx = np.mgrid[0:H, 0:W]

    ox = gx.astype(np.float64) - cx
    oy = gy.astype(np.float64) - cy
    hex_mask = _in_origin_hex(ox, oy, R)
    ix = gx.copy().astype(np.int32)
    iy = gy.copy().astype(np.int32)

    if not hex_mask.any(): return ix, iy, hex_mask

    sx = ox[hex_mask] - float(dx)
    sy = oy[hex_mask] - float(dy)
    wx, wy = _wrap_to_origin_hex(sx, sy, R)

    float_x = wx + cx
    float_y = wy + cy
    src_x = np.clip(np.round(float_x).astype(np.int32), 0, W - 1)
    src_y = np.clip(np.round(float_y).astype(np.int32), 0, H - 1)
    src_x, src_y = _resolve_collisions(src_x, src_y, float_x, float_y, hex_mask, W)

    ix[hex_mask] = src_x
    iy[hex_mask] = src_y
    return ix, iy, hex_mask


def _invert_remap(ix: NDArray[np.int32], iy: NDArray[np.int32], hex_mask: NDArray[np.bool_], H: int, W: int) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    gy, gx = np.mgrid[0:H, 0:W]
    inv_ix = gx.copy().astype(np.int32)
    inv_iy = gy.copy().astype(np.int32)
    ys, xs = np.where(hex_mask)
    inv_ix[iy[ys, xs], ix[ys, xs]] = xs.astype(np.int32)
    inv_iy[iy[ys, xs], ix[ys, xs]] = ys.astype(np.int32)
    return inv_ix, inv_iy


def hex_roll_tensor(tensor: torch.Tensor, dx: int, dy: int, R: float) -> torch.Tensor:
    if dx == 0 and dy == 0: return tensor
    _B, _C, H, W = tensor.shape
    ix, iy, _ = _hex_roll_remap(H, W, dx, dy, R)
    return tensor[
        :, :,
        torch.from_numpy(iy).to(tensor.device).long(),
        torch.from_numpy(ix).to(tensor.device).long(),
    ]


def hex_unroll_tensor(tensor: torch.Tensor, dx: int, dy: int, R: float) -> torch.Tensor:
    if dx == 0 and dy == 0: return tensor
    _B, _C, H, W = tensor.shape
    ix, iy, hex_mask = _hex_roll_remap(H, W, dx, dy, R)
    inv_ix, inv_iy = _invert_remap(ix, iy, hex_mask, H, W)
    return tensor[
        :, :,
        torch.from_numpy(inv_iy).to(tensor.device).long(),
        torch.from_numpy(inv_ix).to(tensor.device).long(),
    ]


def hex_copy_fill_tensor(tensor: torch.Tensor, R: float) -> torch.Tensor:
    _B, _C, H, W = tensor.shape
    cx, cy = W / 2.0, H / 2.0
    gy, gx = np.mgrid[0:H, 0:W]
    ox = gx.astype(np.float64) - cx
    oy = gy.astype(np.float64) - cy
    hex_mask = _in_origin_hex(ox, oy, R)
    outside = ~hex_mask
    if not outside.any(): return tensor

    out_ys, out_xs = np.where(outside)
    ox_o, oy_o = ox[outside], oy[outside]
    wx, wy = _wrap_to_origin_hex(ox_o, oy_o, R)
    copy_src_x = np.clip(np.round(wx + cx).astype(np.int32), 0, W - 1)
    copy_src_y = np.clip(np.round(wy + cy).astype(np.int32), 0, H - 1)
    good = hex_mask[copy_src_y, copy_src_x]
    bad = ~good
    result = tensor.clone()

    if good.any():
        g_dst_y = torch.from_numpy(out_ys[good]).to(tensor.device).long()
        g_dst_x = torch.from_numpy(out_xs[good]).to(tensor.device).long()
        g_src_y = torch.from_numpy(copy_src_y[good]).to(tensor.device).long()
        g_src_x = torch.from_numpy(copy_src_x[good]).to(tensor.device).long()
        result[:, :, g_dst_y, g_dst_x] = tensor[:, :, g_src_y, g_src_x]

    if bad.any():
        bix = copy_src_x[bad].copy()
        biy = copy_src_y[bad].copy()

        still_outside = ~hex_mask[biy, bix]
        if still_outside.any():
            hex_ys, hex_xs = np.where(hex_mask)
            for j in np.where(still_outside)[0]:
                dists = (hex_xs - bix[j])**2 + (hex_ys - biy[j])**2
                nearest = np.argmin(dists)
                bix[j] = hex_xs[nearest]
                biy[j] = hex_ys[nearest]

        b_dst_y = torch.from_numpy(out_ys[bad]).to(tensor.device).long()
        b_dst_x = torch.from_numpy(out_xs[bad]).to(tensor.device).long()
        b_src_y = torch.from_numpy(biy).to(tensor.device).long()
        b_src_x = torch.from_numpy(bix).to(tensor.device).long()
        result[:, :, b_dst_y, b_dst_x] = result[:, :, b_src_y, b_src_x]

    return result


def roll_tensor_mode(tensor: torch.Tensor, dx: int, dy: int, R: float, roll_mode: RollMode) -> torch.Tensor:
    if roll_mode == "hex_copy":
        r = hex_roll_tensor(tensor, dx, dy, R)
        return hex_copy_fill_tensor(r, R)
    elif roll_mode == "hex_copy_no_roll":
        r = hex_roll_tensor(tensor, 0, 0, R)
        return hex_copy_fill_tensor(r, R)
    elif roll_mode == "hex":
        return hex_roll_tensor(tensor, dx, dy, R)
    else:  # roll_mode == "square"
        return torch.roll(tensor, shifts=(dy, dx), dims=(2, 3))


def unroll_tensor_mode(tensor: torch.Tensor, dx: int, dy: int, R: float, roll_mode: RollMode) -> torch.Tensor:
    if roll_mode == "hex_copy":
        u = hex_unroll_tensor(tensor, dx, dy, R)
        return hex_copy_fill_tensor(u, R)
    elif roll_mode == "hex_copy_no_roll":
        return hex_copy_fill_tensor(tensor, R)
    elif roll_mode == "hex":
        return hex_unroll_tensor(tensor, dx, dy, R)
    else:
        return torch.roll(tensor, shifts=(-dy, -dx), dims=(2, 3))

def _random_hex_offset(rng: random.Random, R: float) -> tuple[int, int]:
    hw = np.sqrt(3) / 2.0 * R
    hh = R
    mdx = int(math.floor(hw))
    mdy = int(math.floor(hh))
    if mdx < 1 or mdy < 1:
        return 0, 0
    while True:
        dx = rng.randint(-mdx, mdx)
        dy = rng.randint(-mdy, mdy)
        if dx == 0 and dy == 0:
            continue
        if _in_origin_hex(np.array([float(dx)]), np.array([float(dy)]), R)[0]:
            return dx, dy
