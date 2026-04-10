import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from typing import Callable, Optional, Union

from .geometry import _compute_hex_grid, _sample_nearest, _feather, _hex_sdf, _tile_image_hexagonally
from .constant import SQRT3
from PIL import Image

class HexWrapper:

    img_arr: np.ndarray
    hypotenuse: float
    x_offset: float
    y_offset: float
    outer_margin: float
    inner_padding: float
    gap_padding: float
    feather_width: float
    horizontal_camera_padding: float
    vertical_camera_padding: float
    on_debug: Optional[Callable[[Union[str, np.ndarray]], None]]

    R_base: float
    R_cam: float
    w_cam: float
    h_cam: float
    h_cam_offset: float
    gen_W: int
    gen_H: int
    gen_size: int
    shift_x: float
    shift_y: float
    hcx: float
    hcy: float
    sq_left: int
    sq_top: int
    sq_right: int
    sq_bottom: int
    sq_half_x: float
    sq_half_y: float
    out_W: int
    out_H: int
    d_inward: np.ndarray
    comp_star: np.ndarray
    comp_gap: np.ndarray
    dist_gap: np.ndarray
    wx: np.ndarray
    wy: np.ndarray

    def __init__(
        self,
        img_arr: np.ndarray, hypotenuse: float, x_offset: float, y_offset: float,
        outer_margin: float = 0, inner_padding: float = 0, gap_padding: float = 0, feather_width: float = 0,
        horizontal_camera_padding: float = 0, vertical_camera_padding: float = 0,
        on_debug: Optional[Callable[[Union[str, np.ndarray]], None]] = None,
    ):
        self.img_arr = img_arr
        self.hypotenuse = hypotenuse
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.outer_margin = outer_margin
        self.inner_padding = inner_padding
        self.gap_padding = gap_padding
        self.feather_width = feather_width
        self.horizontal_camera_padding = horizontal_camera_padding
        self.vertical_camera_padding = vertical_camera_padding
        self.on_debug = on_debug

    def wrap(self) -> tuple[np.ndarray, np.ndarray]:
        """Offset image for diffusion. Return (offset_rgb_arr, mask_arr)"""
        img_arr = self.img_arr
        on_debug = self.on_debug

        img_H, img_W = img_arr.shape[:2]
        img_half_x = img_W / 2.0
        img_half_y = img_H / 2.0

        R_base = self.hypotenuse
        R_cam = R_base + (self.outer_margin / SQRT3) if self.outer_margin > 0 else R_base
        r_other = (SQRT3 / 2.0) * R_base
        w_cam = SQRT3 * R_cam
        h_cam = 2.0 * R_cam
        h_cam_offset = 1.5 * R_cam

        shift_x = self.x_offset * w_cam
        shift_y = self.y_offset * h_cam

        sq_half_x = R_cam + self.horizontal_camera_padding
        sq_half_y = R_cam + self.vertical_camera_padding

        # Output dimensions padded to multiple of 8
        raw_W = int(math.ceil(2 * sq_half_x))
        raw_H = int(math.ceil(2 * sq_half_y))
        gen_W = ((raw_W + 7) // 8) * 8
        gen_H = ((raw_H + 7) // 8) * 8

        gy, gx = np.mgrid[0:gen_H, 0:gen_W]
        scale_x = (2.0 * sq_half_x) / gen_W
        scale_y = (2.0 * sq_half_y) / gen_H

        # Shift world coords so that (0,0) is at center of camera instead of top left.
        # It makes the top left coordinate to be (negative, negative) instead of (0, 0)
        wx = (gx.astype(np.float64) + 0.5) * scale_x + (shift_x - sq_half_x)
        wy = (gy.astype(np.float64) + 0.5) * scale_y + (shift_y - sq_half_y)

        if on_debug:
            on_debug("gx")
            on_debug(gx)
            on_debug("wx")
            on_debug(wx)

        min_hex_sdf, min_content_sdf, best_cx, best_cy = _compute_hex_grid(wx, wy, R_cam, r_other, img_half_x, img_half_y)

        # min_hex_sdf: distance to hex edge
        # min_content_sdf: distance to irisan hex & square
        # best_cx, best_cy: which hex center is closest to each pixel
        if on_debug:
            on_debug("min_hex_sdf")
            on_debug(min_hex_sdf.astype(np.uint8))
            on_debug("min_content_sdf")
            on_debug(min_content_sdf.astype(np.uint8))
            on_debug("best_cx")
            on_debug(best_cx.astype(np.uint8))
            on_debug("best_cy")
            on_debug(best_cy.astype(np.uint8))

        d_inward = -min_hex_sdf
        dist_gap = -min_content_sdf

        _ip = self.inner_padding or 0
        _gp = self.gap_padding or 0
        _fw = self.feather_width or 0

        comp_star = _feather(d_inward, _ip, _fw)
        comp_gap = _feather(dist_gap, _gp, _fw)

        # Info for un-offsetting later
        out_W = int(math.ceil(4 * w_cam))
        out_H = int(math.ceil(9 * R_cam))
        hcx, hcy = out_W / 2.0, out_H / 2.0

        sq_left = int(hcx - sq_half_x)
        sq_top = int(hcy - sq_half_y)
        sq_right = int(hcx + sq_half_x)
        sq_bottom = int(hcy + sq_half_y)

        raw_W = sq_right - sq_left
        raw_H = sq_bottom - sq_top
        gy_r, gx_r = np.mgrid[0:raw_H, 0:raw_W]
        wx_r = gx_r.astype(np.float64) + (shift_x - sq_half_x)
        wy_r = gy_r.astype(np.float64) + (shift_y - sq_half_y)

        hsdf_r, csdf_r, bcx_r, bcy_r = _compute_hex_grid(wx_r, wy_r, R_cam, r_other, img_half_x, img_half_y)
        u_r = (wx_r - bcx_r + img_half_x) / img_W
        v_r = (wy_r - bcy_r + img_half_y) / img_H
        valid_r = (u_r >= 0) & (u_r < 1) & (v_r >= 0) & (v_r < 1) & (csdf_r < 0)

        img_arr_rgb = img_arr[..., :3]
        offset_rgb_raw = _sample_nearest(img_arr_rgb, u_r, v_r, valid_r)

        if gen_W != raw_W or gen_H != raw_H:
            offset_rgb_arr = np.array(Image.fromarray(offset_rgb_raw).resize((gen_W, gen_H), Image.Resampling.LANCZOS))
        else:
            offset_rgb_arr = offset_rgb_raw

        mask_arr = (np.maximum(comp_star, comp_gap) * 255).astype(np.uint8)

        if on_debug:
            on_debug("offset_rgb_arr")
            on_debug(offset_rgb_arr)
            on_debug("mask_arr")
            on_debug(mask_arr)

        self.R_base = R_base
        self.R_cam = R_cam
        self.w_cam = w_cam
        self.h_cam = h_cam
        self.h_cam_offset = h_cam_offset
        self.gen_W = gen_W
        self.gen_H = gen_H
        self.gen_size = max(gen_W, gen_H)
        self.shift_x = shift_x
        self.shift_y = shift_y
        self.hcx = hcx
        self.hcy = hcy
        self.sq_left = sq_left
        self.sq_top = sq_top
        self.sq_right = sq_right
        self.sq_bottom = sq_bottom
        self.sq_half_x = sq_half_x
        self.sq_half_y = sq_half_y
        self.out_W = out_W
        self.out_H = out_H
        self.d_inward = d_inward
        self.comp_star = comp_star
        self.comp_gap = comp_gap
        self.dist_gap = dist_gap
        self.wx = wx
        self.wy = wy

        return offset_rgb_arr, mask_arr


    def debug_wrap(self, offset_rgb_arr: np.ndarray, mask_arr: np.ndarray, overlay_linewidth: float = 2.0) -> np.ndarray:
        gen_W = self.gen_W
        gen_H = self.gen_H
        outer_margin = self.outer_margin
        inner_padding = self.inner_padding
        gap_padding = self.gap_padding
        feather_width = self.feather_width
        d_inward = self.d_inward
        comp_star = self.comp_star
        comp_gap = self.comp_gap
        dist_gap = self.dist_gap
        wx = self.wx
        wy = self.wy
        shift_x = self.shift_x
        shift_y = self.shift_y
        R_cam = self.R_cam

        dpi = 100
        fig = plt.figure(figsize=(gen_W/dpi, gen_H/dpi), dpi=dpi)
        fig.patch.set_visible(False)
        ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
        ax.axis("off")
        ax.set_xlim(0, gen_W)
        ax.set_ylim(gen_H, 0)

        ax.imshow(offset_rgb_arr)

        overlay_rgba = np.zeros((gen_H, gen_W, 4), dtype=np.uint8)
        overlay_rgba[..., 0] = 255
        overlay_rgba[..., 3] = (mask_arr * 0.5).astype(np.uint8)
        ax.imshow(overlay_rgba, alpha=0.5)

        x_coords = np.arange(gen_W)
        y_coords = np.arange(gen_H)

        if outer_margin > 0:
            ax.contour(x_coords, y_coords, d_inward, levels=[0.0], colors=["magenta"], linewidths=overlay_linewidth)

        if inner_padding > 0:
            ax.contour(x_coords, y_coords, d_inward, levels=[inner_padding], colors=["orange"], linewidths=overlay_linewidth)

        if gap_padding > 0:
            ax.contour(x_coords, y_coords, dist_gap, levels=[gap_padding], colors=["cyan"], linewidths=overlay_linewidth)

        if feather_width > 0 and (inner_padding > 0 or gap_padding > 0):
            mask = np.maximum(comp_star, comp_gap) * 255
            ax.contour(x_coords, y_coords, mask, levels=[1.0], colors=["yellow"], linewidths=overlay_linewidth)

        lx_cam = wx - shift_x
        ly_cam = wy - shift_y
        R_cam_other = (SQRT3 / 2.0) * R_cam
        cam_hex_sdf = _hex_sdf(lx_cam, ly_cam, R_cam_other)
        ax.contour(x_coords, y_coords, cam_hex_sdf, levels=[0.0], colors=["lime"], linewidths=overlay_linewidth)

        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        debug_arr = np.asarray(buf).copy()
        plt.close(fig)
        return debug_arr

    
    def unwrap(self, inpainted_rgb: np.ndarray, output_size: int = 0) -> tuple[np.ndarray, float]:
        R_cam = self.R_cam
        w_cam = self.w_cam
        h_cam = self.h_cam

        gen_W, gen_H = self.gen_W, self.gen_H
        raw_W = self.sq_right - self.sq_left
        raw_H = self.sq_bottom - self.sq_top
        sc_x = gen_W / raw_W if raw_W > 0 else 1.0
        sc_y = gen_H / raw_H if raw_H > 0 else 1.0
        sc = (sc_x + sc_y) / 2.0 # will be used for scaling R_cam and crop_side

        inp_H, inp_W = inpainted_rgb.shape[:2]
        if inp_W != gen_W or inp_H != gen_H:
            inpainted_rgb = _resize_nearest(inpainted_rgb, gen_W, gen_H)

        Rcs = R_cam * sc

        pad_raw = int(math.ceil(2 * R_cam))
        pad_gen = ((pad_raw + 7) // 8) * 8
        crop_side = int(math.ceil(pad_gen * sc))

        gy, gx = np.mgrid[0:gen_H, 0:gen_W]
        hx = gx.astype(np.float64) + 0.5 - gen_W / 2.0
        hy = gy.astype(np.float64) + 0.5 - gen_H / 2.0
        r_inscribed = (SQRT3 / 2.0) * Rcs
        hex_mask = _hex_sdf(hx, hy, r_inscribed) < 0

        tile = np.zeros((gen_H, gen_W, 3), dtype=np.uint8)
        tile[hex_mask] = inpainted_rgb[hex_mask, :3]

        hcx_gen = (self.sq_half_x - self.x_offset * w_cam) * sc_x
        hcy_gen = (self.sq_half_y - self.y_offset * h_cam) * sc_y
        offset_x = crop_side / 2.0 - (hcx_gen - gen_W / 2.0)
        offset_y = crop_side / 2.0 - (hcy_gen - gen_H / 2.0)

        result_arr = _tile_image_hexagonally(tile, crop_side, crop_side, Rcs, offset_x, offset_y)

        if output_size > 0 and output_size != crop_side:
            result_arr = _resize_nearest(result_arr, output_size, output_size)

        final_size = output_size if output_size > 0 else crop_side
        R_final = R_cam * final_size / pad_gen
        return result_arr, R_final
    

def _resize_nearest(arr: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    in_h, in_w = arr.shape[:2]
    ry, rx = np.mgrid[0:out_h, 0:out_w]
    sy = np.clip(ry * in_h // out_h, 0, in_h - 1)
    sx = np.clip(rx * in_w // out_w, 0, in_w - 1)
    return arr[sy, sx]