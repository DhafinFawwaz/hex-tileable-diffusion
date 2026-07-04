
from PIL import Image
import numpy as np
import torch

from hex_tileable_diffusion.types import ImageInput

def load_image(image: ImageInput, output_size: int):
    if isinstance(image, str): image = Image.open(image)
    if isinstance(image, Image.Image): image = image.convert("RGB")

    elif isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[2] in [1, 3, 4]:
            if image.dtype != np.uint8:
                image = (image * 255).clip(0, 255).astype(np.uint8)
            image = Image.fromarray(image)
        else:
            raise ValueError("NumPy array must be HWC format")

    elif isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
        if image.ndim == 3 and image.shape[0] in [1, 3]:
            image = np.transpose(image, (1, 2, 0))
        image = (image * 255).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(image)

    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    w, h = image.size
    if w != h:
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        image = image.crop((left, top, left + side, top + side))

    image = image.resize((output_size, output_size), Image.Resampling.LANCZOS)
    image = np.array(image)

    return image