"""
Gozumol - AI-Powered Visual Assistance for the Visually Impaired

Gozumol is an open-source library designed to help visually impaired individuals
navigate their daily lives more safely and independently using AI-powered vision.
"""

__version__ = "0.1.0"
__author__ = "Gozumol Contributors"

from .core.assistant import VisionAssistant
from .core.model import load_model, load_processor

__all__ = [
    "VisionAssistant",
    "load_model",
    "load_processor",
    "__version__",
]
