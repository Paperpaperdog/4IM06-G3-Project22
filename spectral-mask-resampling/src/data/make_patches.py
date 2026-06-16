import random

from PIL import Image


def random_crop_rgb(img: Image.Image, patch_size: int, rng: random.Random) -> Image.Image:
    img = img.convert("RGB")
    width, height = img.size
    if width < patch_size or height < patch_size:
        raise ValueError(f"Image is smaller than patch size: {width}x{height} < {patch_size}")
    left = rng.randint(0, width - patch_size)
    top = rng.randint(0, height - patch_size)
    return img.crop((left, top, left + patch_size, top + patch_size))
