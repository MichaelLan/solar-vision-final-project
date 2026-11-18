from io import BytesIO

import numpy as np
from PIL import Image, UnidentifiedImageError

from solar_ml.constants import HEIGHT, WIDTH


def decode_pixels(content: bytes) -> np.ndarray:
    try:
        with Image.open(BytesIO(content)) as image:
            if image.size != (WIDTH, HEIGHT):
                raise ValueError(f"Expected {WIDTH}x{HEIGHT}, got {image.size}")
            rgb = np.asarray(image.convert("RGB"), dtype=np.float64)
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Invalid image") from exc
    luminance = np.floor(0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2] + 0.5)
    return (luminance / 255.0).astype(np.float32).reshape(-1)


def perceptual_hash(pixels: np.ndarray) -> int:
    image = Image.fromarray(np.rint(pixels.reshape(HEIGHT, WIDTH) * 255).astype(np.uint8))
    small = np.asarray(image.resize((9, 8), Image.Resampling.BILINEAR))
    bits = small[:, 1:] > small[:, :-1]
    return int.from_bytes(np.packbits(bits.reshape(-1)).tobytes(), "big")
