from PIL import Image
from IPython.display import display
from hex_tileable_diffusion.core.wrapper import wrap_hexagon_image, debug_wrap_hexagon_image_info

import time

image_path = "demos/rock1_512.png"
output_dir = "."
output_size = 512

import numpy as np

input_image = (
    Image.open(image_path)
    .convert("RGBA")
    .resize((output_size, output_size), Image.Resampling.LANCZOS)
)
input_arr = np.array(input_image)

t0 = time.time()

rgb_arr, mask_arr, info = wrap_hexagon_image(
    img_arr=input_arr,
    hypotenuse=306,
    x_offset=0,
    y_offset=0.5,
    outer_margin=75,
    inner_padding=75,
    gap_padding=50,
    feather_width=30,
    horizontal_camera_padding=540,
    vertical_camera_padding=560,
    show_debug=True,
)

t1 = time.time()
print(f"{t1-t0:.3f}s")

outer_margin = info.outer_margin
inner_padding = info.inner_padding
gap_padding = info.gap_padding
feather_width = info.feather_width
R_cam = info.R_cam
hypotenuse = info.R_base
horizontal_camera_padding = info.horizontal_camera_padding
vertical_camera_padding = info.vertical_camera_padding
(gen_W, gen_H) = (info.gen_W, info.gen_H)
R_cam = info.R_cam
(shift_x, shift_y) = (info.shift_x, info.shift_y)

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

debug_arr = debug_wrap_hexagon_image_info(info)
display(Image.fromarray(debug_arr))


Image.fromarray(debug_arr).save("demos/rock1_wrapped_output.png")
Image.fromarray(rgb_arr).save("demos/rock1_wrapped_rgb.png")