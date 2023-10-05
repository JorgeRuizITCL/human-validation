
from typing import Tuple

import numpy as np
from PIL import Image


def to_square(pil_img: Image.Image, background_color: Tuple[int, int, int]):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def resize_pad_crop(crop: np.ndarray, resolution: Tuple[int, int]) -> np.ndarray:
    base_img = Image.fromarray(crop)
    squared_img = to_square(base_img, (0,0,0))
    resized_img = squared_img.resize(resolution)    
    return np.array(resized_img)