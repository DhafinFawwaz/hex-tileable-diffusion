
import numpy as np
from dataclasses import dataclass

@dataclass
class HexWrapInfo:

    # Internal parameters
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
    x_offset: float
    y_offset: float
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
    horizontal_camera_padding: float
    vertical_camera_padding: float
    
    # Config 
    outer_margin: float
    inner_padding: float
    gap_padding: float
    feather_width: float
    
    # Output arrays
    offset_rgb_arr: np.ndarray   # (H, W, 3) uint8
    mask_arr: np.ndarray         # (H, W) uint8
    
    # Debug arrays
    d_inward: np.ndarray
    comp_star: np.ndarray
    comp_gap: np.ndarray
    dist_gap: np.ndarray
    wx: np.ndarray
    wy: np.ndarray