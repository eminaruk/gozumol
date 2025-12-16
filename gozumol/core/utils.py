"""
Gozumol Utility Functions

This module contains utility functions used throughout the library,
including device detection, timing helpers, and image processing utilities.
"""

import time
from typing import Dict, Optional, Union
from pathlib import Path

import torch
from PIL import Image


def get_device(preferred_device: Optional[str] = None) -> str:
    """
    Determine the best available device for model inference.

    Args:
        preferred_device: Optional preferred device ("cuda", "cpu", or "auto").
                         If None or "auto", automatically selects the best available.

    Returns:
        Device string for PyTorch ("cuda" or "cpu")
    """
    if preferred_device and preferred_device != "auto":
        if preferred_device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            return "cpu"
        return preferred_device

    # Auto-detect best device
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_device_info() -> Dict[str, Union[str, int, float]]:
    """
    Get detailed information about the current device.

    Returns:
        Dictionary containing device information
    """
    info = {
        "device": "cpu",
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info["device"] = "cuda"
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_memory_total_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        info["cuda_memory_allocated_gb"] = torch.cuda.memory_allocated(0) / (1024**3)
        info["cuda_memory_cached_gb"] = torch.cuda.memory_reserved(0) / (1024**3)

    return info


def format_timing_info(timing_info: Dict[str, float]) -> str:
    """
    Format timing information into a readable string.

    Args:
        timing_info: Dictionary containing timing measurements

    Returns:
        Formatted string with timing breakdown
    """
    lines = [
        "Timing Information:",
        f"  - Processor time: {timing_info.get('processor_time', 0):.4f}s",
        f"  - Generation time: {timing_info.get('generation_time', 0):.4f}s",
        f"  - Decode time: {timing_info.get('decode_time', 0):.4f}s",
        f"  - Total time: {timing_info.get('total_time', 0):.4f}s",
    ]
    return "\n".join(lines)


def load_image(image_source: Union[str, Path, Image.Image]) -> Image.Image:
    """
    Load an image from various sources.

    Args:
        image_source: Path to image file, URL, or PIL Image object

    Returns:
        PIL Image object
    """
    if isinstance(image_source, Image.Image):
        return image_source

    image_path = Path(image_source) if isinstance(image_source, str) else image_source

    if image_path.exists():
        return Image.open(image_path)

    # Try to load from URL
    if isinstance(image_source, str) and image_source.startswith(("http://", "https://")):
        import requests
        response = requests.get(image_source, stream=True, timeout=10)
        response.raise_for_status()
        return Image.open(response.raw)

    raise ValueError(f"Could not load image from: {image_source}")


def resize_image_if_needed(
    image: Image.Image,
    max_size: int = 1024,
    maintain_aspect_ratio: bool = True
) -> Image.Image:
    """
    Resize image if it exceeds maximum dimensions.

    Args:
        image: PIL Image to resize
        max_size: Maximum dimension (width or height)
        maintain_aspect_ratio: Whether to maintain aspect ratio

    Returns:
        Resized PIL Image
    """
    width, height = image.size

    if width <= max_size and height <= max_size:
        return image

    if maintain_aspect_ratio:
        ratio = min(max_size / width, max_size / height)
        new_size = (int(width * ratio), int(height * ratio))
    else:
        new_size = (max_size, max_size)

    return image.resize(new_size, Image.Resampling.LANCZOS)


class Timer:
    """
    Simple context manager for timing code blocks.

    Usage:
        with Timer() as t:
            # code to time
        print(f"Elapsed: {t.elapsed:.4f}s")
    """

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.elapsed = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time


def clear_gpu_memory():
    """
    Clear GPU memory cache to free up VRAM.
    Useful when switching between models or after intensive operations.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
