import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from IPython.display import display
from matplotlib.backends.backend_agg import FigureCanvasAgg

from .info import HexWrapInfo
from .geometry import _compute_hex_grid, _sample_nearest, _feather, _hex_sdf, _pixel_to_hex, _cube_round, _hex_to_pixel
from .constant import SQRT3

def wrap_hexagon_image(
    img_arr: np.ndarray, hypotenuse: float, x_offset: float, y_offset: float,
    outer_margin: float = 0, inner_padding: float = 0, gap_padding: float = 0, feather_width: float = 0,
    horizontal_camera_padding: float = 0, vertical_camera_padding: float = 0,
    show_debug: bool = True,
) -> tuple[np.ndarray, np.ndarray, HexWrapInfo]:
    img_H, img_W = img_arr.shape[:2]
    img_half_x = img_W / 2.0
    img_half_y = img_H / 2.0

    R_base = hypotenuse
    R_cam = R_base + (outer_margin / SQRT3) if outer_margin > 0 else R_base
    r_other = (SQRT3 / 2.0) * R_base
    w_cam = SQRT3 * R_cam
    h_cam = 2.0 * R_cam
    h_cam_offset = 1.5 * R_cam

    shift_x = x_offset * w_cam
    shift_y = y_offset * h_cam

    sq_half_x = R_cam + horizontal_camera_padding
    sq_half_y = R_cam + vertical_camera_padding

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

    if show_debug:
        print("gx")
        print(gx)
        print("wx")
        print(wx)

    min_hex_sdf, min_content_sdf, best_cx, best_cy = _compute_hex_grid(wx, wy, R_cam, r_other, img_half_x, img_half_y)

    # min_hex_sdf: distance to hex edge
    # min_content_sdf: distance to irisan hex & square
    # best_cx, best_cy: which hex center is closest to each pixel
    if show_debug: 
        print("min_hex_sdf")
        display(Image.fromarray(min_hex_sdf.astype(np.uint8)))
        print("min_content_sdf")
        display(Image.fromarray(min_content_sdf.astype(np.uint8)))
        print("best_cx")
        display(Image.fromarray(best_cx.astype(np.uint8)))
        print("best_cy")
        display(Image.fromarray(best_cy.astype(np.uint8)))

    d_inward = -min_hex_sdf
    dist_gap = -min_content_sdf

    lx = wx - best_cx
    ly = wy - best_cy

    # Convert to UV for sampling the input image
    # UV is just UV coords
    u = (lx + img_half_x) / img_W
    v = (ly + img_half_y) / img_H
    valid = (u >= 0) & (u < 1) & (v >= 0) & (v < 1) & (min_content_sdf < 0)

    img_arr_rgb = img_arr[..., :3] # exclude alpha
    offset_rgb_arr = _sample_nearest(img_arr_rgb, u, v, valid)

    if show_debug:
        print("offset_rgb_arr")
        display(Image.fromarray(offset_rgb_arr))

    _ip = inner_padding or 0
    _gp = gap_padding or 0
    _fw = feather_width or 0

    comp_star = _feather(d_inward, _ip, _fw)
    comp_gap = _feather(dist_gap, _gp, _fw)
    mask_arr = (np.maximum(comp_star, comp_gap) * 255).astype(np.uint8)

    if show_debug:
        print("mask_arr")
        display(Image.fromarray(mask_arr, mode="L"))

    if show_debug:
        print("offset_rgb_arr")
        display(Image.fromarray(offset_rgb_arr))

    # Info for un-offsetting later
    out_W = int(math.ceil(4 * w_cam))
    out_H = int(math.ceil(9 * R_cam))
    hcx, hcy = out_W / 2.0, out_H / 2.0

    sq_left = int(hcx - sq_half_x)
    sq_top = int(hcy - sq_half_y)
    sq_right = int(hcx + sq_half_x)
    sq_bottom = int(hcy + sq_half_y)

    info = HexWrapInfo(
        R_base=R_base, R_cam=R_cam, w_cam=w_cam, h_cam=h_cam,
        h_cam_offset=h_cam_offset,
        gen_W=gen_W, gen_H=gen_H,
        gen_size=max(gen_W, gen_H),
        shift_x=shift_x, shift_y=shift_y,
        x_offset=x_offset, y_offset=y_offset,
        hcx=hcx, hcy=hcy,
        sq_left=sq_left, sq_top=sq_top,
        sq_right=sq_right, sq_bottom=sq_bottom,
        sq_half_x=sq_half_x, sq_half_y=sq_half_y,
        out_W=out_W, out_H=out_H,
        horizontal_camera_padding=horizontal_camera_padding,
        vertical_camera_padding=vertical_camera_padding,
        outer_margin=outer_margin,
        inner_padding=_ip,
        gap_padding=_gp,
        feather_width=_fw,
        offset_rgb_arr=offset_rgb_arr,
        mask_arr=mask_arr,
        d_inward=d_inward,
        comp_star=comp_star,
        comp_gap=comp_gap,
        dist_gap=dist_gap,
        wx=wx,
        wy=wy,
    )

    return offset_rgb_arr, mask_arr, info


def debug_wrap_hexagon_image_info(info: HexWrapInfo, overlay_linewidth: float = 2.0):
    gen_W = info.gen_W
    gen_H = info.gen_H
    offset_rgb_arr = info.offset_rgb_arr
    mask_arr = info.mask_arr
    outer_margin = info.outer_margin
    inner_padding = info.inner_padding
    gap_padding = info.gap_padding
    feather_width = info.feather_width
    d_inward = info.d_inward
    comp_star = info.comp_star
    comp_gap = info.comp_gap
    dist_gap = info.dist_gap
    wx = info.wx
    wy = info.wy
    shift_x = info.shift_x
    shift_y = info.shift_y
    R_cam = info.R_cam

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


def tile_image_hexagonally(tile_arr: np.ndarray, out_w: int, out_h: int, R: float, offset_x: float = 0.0, offset_y: float = 0.0) -> np.ndarray:
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
