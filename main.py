from PIL import Image
from IPython.display import display, Image as IPImage
from hex_tileable_diffusion.core.hexwrapper import HexWrapper
from torch import cuda

import time
import numpy as np

def debug_display(v, scale=0.25):
    if isinstance(v, str):
        print(v)
    elif isinstance(v, np.ndarray) and v.dtype == np.uint8:
        img = Image.fromarray(v)
        w = img.size[0]
        display(IPImage(data=img._repr_png_(), width=int(w * scale)))
    else:
        print(v)

def get_machine_info() -> tuple[str, str | None, float | None]:
    if cuda.is_available():
        return ((
            "cuda",
            cuda.get_device_name(0),
            cuda.get_device_properties(0).total_memory / (1024 ** 3),
        ))
    else:
        return (("cpu", None, None)) 

print("Machine Info:", get_machine_info())

image_path = "demos/rock1_512.png"
output_dir = "."
output_size = 512

input_image = (
    Image.open(image_path)
    .convert("RGBA")
    .resize((output_size, output_size), Image.Resampling.LANCZOS)
)
input_arr = np.array(input_image)

t0 = time.time()

hex_wrapper = HexWrapper(
    hypotenuse=306,
    x_offset=0,
    y_offset=0.5,
    outer_margin=75,
    inner_padding=75,
    gap_padding=50,
    feather_width=30,
    horizontal_camera_padding=540,
    vertical_camera_padding=560,
    on_debug=debug_display,
)
rgb_arr, mask_arr, wrap_debug = hex_wrapper.wrap(input_arr)

t1 = time.time()
print(f"{t1-t0:.3f}s")

outer_margin = hex_wrapper.outer_margin
inner_padding = hex_wrapper.inner_padding
gap_padding = hex_wrapper.gap_padding
feather_width = hex_wrapper.feather_width
R_cam = hex_wrapper.R_cam
hypotenuse = hex_wrapper.R_base
horizontal_camera_padding = hex_wrapper.horizontal_camera_padding
vertical_camera_padding = hex_wrapper.vertical_camera_padding
(gen_W, gen_H) = (hex_wrapper.gen_W, hex_wrapper.gen_H)
R_cam = hex_wrapper.R_cam
(shift_x, shift_y) = (hex_wrapper.shift_x, hex_wrapper.shift_y)

print("Camera View")
print("Outer Margin (Magenta):", outer_margin)
print("Inner Padding (Orange):", inner_padding)
print("Gap Padding (Cyan):", gap_padding)
print("Feather Width (Yellow):", feather_width)
print("Camera Hex (Green):", R_cam)
print("Hypotenuse:", hypotenuse)
print("Horizontal Camera Padding:", horizontal_camera_padding)
print("Vertical Camera Padding:", vertical_camera_padding)
print(f"(Generated Width, Generated Height): ({gen_W}, {gen_H})")
print(f"(Shift X, Shift Y): ({shift_x}, {shift_y})")
print(f"R_cam: {R_cam}")

debug_arr = hex_wrapper.debug_wrap(rgb_arr, mask_arr, wrap_debug)
debug_display(debug_arr)

Image.fromarray(debug_arr).save("demos/rock1_wrapped_output.png")
Image.fromarray(rgb_arr).save("demos/rock1_wrapped_rgb.png")