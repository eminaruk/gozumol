"""
Gozumol Vision Assistant

This module contains the main VisionAssistant class that provides
a high-level interface for visual assistance functionality.
"""

from typing import Dict, List, Optional, Union
from pathlib import Path

import torch
from PIL import Image

from .model import (
    load_model,
    load_processor,
    load_generation_config,
    generate_response,
    DEFAULT_MODEL_ID,
)
from .utils import get_device, load_image, resize_image_if_needed, format_timing_info
from ..prompts import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_PROMPT,
    get_prompt_pair,
)


class VisionAssistant:
    """
    AI-powered visual assistant for visually impaired users.

    This class provides a high-level interface for analyzing images and
    providing navigation guidance to visually impaired users.

    Attributes:
        model: The loaded Phi-4 multimodal model
        processor: The model processor
        generation_config: Configuration for text generation
        system_prompt: Current system prompt defining assistant behavior
        user_prompt: Current user prompt for image analysis
        device: Device being used for inference

    Example:
        >>> from gozumol import VisionAssistant
        >>> assistant = VisionAssistant()
        >>> description = assistant.describe("path/to/image.jpg")
        >>> print(description)
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: Optional[str] = None,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        scenario: str = "default",
        max_new_tokens: int = 1024,
        low_memory_mode: bool = False,
        lazy_load: bool = False
    ):
        """
        Initialize the Vision Assistant.

        Args:
            model_id: HuggingFace model identifier
            device: Target device ("cuda", "cpu", or "auto")
            system_prompt: Custom system prompt (overrides scenario)
            user_prompt: Custom user prompt (overrides scenario)
            scenario: Prompt scenario ("default", "outdoor", "indoor", etc.)
            max_new_tokens: Maximum tokens to generate
            low_memory_mode: Enable memory optimizations
            lazy_load: If True, defer model loading until first use
        """
        self.model_id = model_id
        self.device = get_device(device)
        self.max_new_tokens = max_new_tokens
        self.low_memory_mode = low_memory_mode

        # Set prompts based on scenario or custom values
        if system_prompt is not None or user_prompt is not None:
            self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
            self.user_prompt = user_prompt or DEFAULT_USER_PROMPT
        else:
            self.system_prompt, self.user_prompt = get_prompt_pair(scenario)

        # Model components (loaded lazily if requested)
        self._model = None
        self._processor = None
        self._generation_config = None

        if not lazy_load:
            self._load_model()

    def _load_model(self):
        """Load the model, processor, and generation config."""
        print(f"Loading model on {self.device}...")

        self._processor = load_processor(self.model_id)
        self._model = load_model(
            model_id=self.model_id,
            device=self.device,
            low_memory_mode=self.low_memory_mode,
        )
        self._generation_config = load_generation_config(
            self.model_id,
            max_new_tokens=self.max_new_tokens
        )

        print("Model loaded successfully!")

    @property
    def model(self):
        """Get the model, loading it if necessary."""
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def processor(self):
        """Get the processor, loading it if necessary."""
        if self._processor is None:
            self._load_model()
        return self._processor

    @property
    def generation_config(self):
        """Get the generation config, loading it if necessary."""
        if self._generation_config is None:
            self._load_model()
        return self._generation_config

    def describe(
        self,
        image: Union[str, Path, Image.Image],
        user_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        return_timing: bool = False,
        max_image_size: int = 1024
    ) -> Union[str, tuple]:
        """
        Analyze an image and return a navigation description.

        Args:
            image: Image to analyze (path, URL, or PIL Image)
            user_prompt: Override the default user prompt for this request
            system_prompt: Override the default system prompt for this request
            return_timing: If True, also return timing information
            max_image_size: Maximum image dimension (resized if larger)

        Returns:
            Description string, or tuple of (description, timing_info) if return_timing=True
        """
        # Load and prepare image
        img = load_image(image)
        img = resize_image_if_needed(img, max_image_size)

        # Use provided prompts or defaults
        sys_prompt = system_prompt or self.system_prompt
        usr_prompt = user_prompt or self.user_prompt

        # Build content list
        content_list = [
            {"type": "image", "content": img, "role": "user"},
            {"type": "text", "content": usr_prompt, "role": "user"},
        ]

        # Generate response
        _, response, timing_info = generate_response(
            model=self.model,
            processor=self.processor,
            system_prompt=sys_prompt,
            content_list=content_list,
            generation_config=self.generation_config,
            max_new_tokens=self.max_new_tokens,
            device=self.device,
        )

        if return_timing:
            return response, timing_info
        return response

    def quick_scan(
        self,
        image: Union[str, Path, Image.Image],
        return_timing: bool = False
    ) -> Union[str, tuple]:
        """
        Perform a quick safety scan of the environment.

        This method uses a condensed prompt focused on immediate
        safety concerns and obstacles.

        Args:
            image: Image to analyze
            return_timing: If True, also return timing information

        Returns:
            Brief safety assessment
        """
        from ..prompts import QUICK_SCAN_USER_PROMPT

        return self.describe(
            image,
            user_prompt=QUICK_SCAN_USER_PROMPT,
            return_timing=return_timing
        )

    def check_crossing(
        self,
        image: Union[str, Path, Image.Image],
        return_timing: bool = False
    ) -> Union[str, tuple]:
        """
        Check if it's safe to cross a street or intersection.

        This method uses prompts optimized for traffic safety
        and crossing assessment.

        Args:
            image: Image to analyze
            return_timing: If True, also return timing information

        Returns:
            Crossing safety assessment with clear guidance
        """
        from ..prompts import (
            TRAFFIC_SAFETY_SYSTEM_PROMPT,
            CROSSING_ASSISTANCE_USER_PROMPT
        )

        return self.describe(
            image,
            system_prompt=TRAFFIC_SAFETY_SYSTEM_PROMPT,
            user_prompt=CROSSING_ASSISTANCE_USER_PROMPT,
            return_timing=return_timing
        )

    def set_scenario(self, scenario: str):
        """
        Change the current prompt scenario.

        Args:
            scenario: New scenario ("default", "outdoor", "indoor",
                     "traffic", "crowd", "quick", "detailed", "safety", "crossing")
        """
        self.system_prompt, self.user_prompt = get_prompt_pair(scenario)

    def set_prompts(
        self,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None
    ):
        """
        Set custom prompts.

        Args:
            system_prompt: New system prompt
            user_prompt: New user prompt
        """
        if system_prompt is not None:
            self.system_prompt = system_prompt
        if user_prompt is not None:
            self.user_prompt = user_prompt

    def get_device_info(self) -> Dict:
        """
        Get information about the current device configuration.

        Returns:
            Dictionary with device details
        """
        from .utils import get_device_info
        return get_device_info()

    def clear_memory(self):
        """
        Clear GPU memory cache.

        Useful after processing many images or when switching scenarios.
        """
        from .utils import clear_gpu_memory
        clear_gpu_memory()

    def __repr__(self) -> str:
        return (
            f"VisionAssistant("
            f"model='{self.model_id}', "
            f"device='{self.device}', "
            f"loaded={self._model is not None})"
        )
