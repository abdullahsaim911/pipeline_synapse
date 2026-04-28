"""
Image Pre-scaling Utility for VLM Processing.

Downscales frames to max 768x768 before VLM processing to reduce vision token count
and prevent KV cache overflow on 6GB VRAM.

"""

from PIL import Image
from pathlib import Path
from typing import Union
import io

# Maximum resolution for VLM input (increased from 768x768 for better OCR quality)
MAX_RESOLUTION = (1024, 1024)


def preprocess_image(
    image_path_or_pil: Union[str, Path, Image.Image],
    max_resolution: tuple = MAX_RESOLUTION,
    maintain_aspect_ratio: bool = True
) -> Image.Image:
    """
    Preprocess image for VLM by downscaling if needed.

    Args:
        image_path_or_pil: Path to image or PIL Image object
        max_resolution: Maximum (width, height) tuple (default: 768x768)
        maintain_aspect_ratio: If True, maintain aspect ratio while fitting within bounds

    Returns:
        PIL Image object (RGB mode)
    """
    # Load image if path provided
    if isinstance(image_path_or_pil, (str, Path)):
        image = Image.open(image_path_or_pil).convert("RGB")
    else:
        image = image_path_or_pil.convert("RGB")

    original_size = image.size
    width, height = original_size

    # Check if scaling is needed
    if width <= max_resolution[0] and height <= max_resolution[1]:
        return image  # No scaling needed

    if maintain_aspect_ratio:
        # Calculate scaling factor to fit within bounds
        scale_factor = min(
            max_resolution[0] / width,
            max_resolution[1] / height
        )
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        image = image.resize((new_width, new_height), Image.LANCZOS)
    else:
        # Direct resize to max resolution (may distort)
        image = image.resize(max_resolution, Image.LANCZOS)

    print(f"[Image Preprocessor] Scaled: {original_size} -> {image.size}")

    return image


def preprocess_image_to_bytes(
    image_path_or_pil: Union[str, Path, Image.Image],
    format: str = "JPEG",
    quality: int = 85
) -> bytes:
    """
    Preprocess image and return as bytes for VLM input.

    Args:
        image_path_or_pil: Path to image or PIL Image object
        format: Output format (JPEG, PNG, etc.)
        quality: JPEG quality (1-100)

    Returns:
        Image bytes
    """
    image = preprocess_image(image_path_or_pil)

    buffer = io.BytesIO()
    image.save(buffer, format=format, quality=quality)
    buffer.seek(0)

    return buffer.getvalue()
