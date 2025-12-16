"""
Gozumol Model Module

This module handles loading and configuring the Phi-4 Multimodal model
for visual assistance. It supports both CPU and GPU inference.
"""

import time
from typing import Dict, List, Optional, Tuple, Union

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

from .utils import get_device, Timer


# Default model configuration
DEFAULT_MODEL_ID = "microsoft/Phi-4-multimodal-instruct"


def load_processor(
    model_id: str = DEFAULT_MODEL_ID,
    trust_remote_code: bool = True
) -> AutoProcessor:
    """
    Load the processor for the Phi-4 multimodal model.

    Args:
        model_id: HuggingFace model identifier
        trust_remote_code: Whether to trust remote code execution

    Returns:
        Loaded AutoProcessor instance
    """
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code
    )
    return processor


def load_model(
    model_id: str = DEFAULT_MODEL_ID,
    device: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
    use_flash_attention: bool = True,
    trust_remote_code: bool = True,
    low_memory_mode: bool = False
) -> AutoModelForCausalLM:
    """
    Load the Phi-4 multimodal model for inference.

    Args:
        model_id: HuggingFace model identifier
        device: Target device ("cuda", "cpu", or "auto")
        torch_dtype: Data type for model weights (auto-detected if None)
        use_flash_attention: Whether to use Flash Attention 2 (GPU only)
        trust_remote_code: Whether to trust remote code execution
        low_memory_mode: Enable memory optimizations for limited RAM/VRAM

    Returns:
        Loaded model instance
    """
    device = get_device(device)

    # Determine optimal dtype
    if torch_dtype is None:
        if device == "cuda":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

    # Configure attention implementation
    attn_implementation = None
    if use_flash_attention and device == "cuda":
        try:
            import flash_attn
            attn_implementation = "flash_attention_2"
        except ImportError:
            print("Flash Attention not available. Using default attention.")

    # Configure device map
    if device == "cuda":
        device_map = "cuda"
    else:
        device_map = "cpu"

    # Additional kwargs for memory optimization
    model_kwargs = {
        "pretrained_model_name_or_path": model_id,
        "device_map": device_map,
        "torch_dtype": torch_dtype,
        "trust_remote_code": trust_remote_code,
    }

    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    if low_memory_mode:
        model_kwargs["low_cpu_mem_usage"] = True

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)

    # Move to device if needed
    if device == "cuda" and not str(next(model.parameters()).device).startswith("cuda"):
        model = model.cuda()

    return model


def load_generation_config(
    model_id: str = DEFAULT_MODEL_ID,
    max_new_tokens: int = 1024,
    **kwargs
) -> GenerationConfig:
    """
    Load and configure generation settings.

    Args:
        model_id: HuggingFace model identifier
        max_new_tokens: Maximum number of tokens to generate
        **kwargs: Additional generation config parameters

    Returns:
        GenerationConfig instance
    """
    config = GenerationConfig.from_pretrained(model_id)

    # Override with custom settings
    config.max_new_tokens = max_new_tokens

    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config


def build_prompt(
    system_prompt: str,
    content_list: List[Dict],
) -> Tuple[str, List[Image.Image], List[tuple]]:
    """
    Build the complete prompt from system message and content items.

    Args:
        system_prompt: System message defining assistant behavior
        content_list: List of content items with type, content, and role

    Returns:
        Tuple of (formatted_prompt, images_list, audios_list)
    """
    # Token definitions
    system_token = "<|system|>"
    user_token = "<|user|>"
    assistant_token = "<|assistant|>"
    end_token = "<|end|>"

    # Initialize prompt with system message
    complete_prompt = f"{system_token}{system_prompt}{end_token}"

    # Collect media
    images = []
    audios = []

    # Build conversation
    current_role = None
    role_content = ""

    for item in content_list:
        item_type = item["type"]
        item_role = item.get("role", "user")

        # Handle role transitions
        if current_role is not None and current_role != item_role:
            role_token = user_token if current_role == "user" else assistant_token
            complete_prompt += f"{role_token}{role_content}{end_token}"
            role_content = ""

        current_role = item_role

        # Process content types
        if item_type == "text":
            role_content += item["content"]

        elif item_type == "image":
            image_index = len(images) + 1
            role_content += f"<|image_{image_index}|>"
            images.append(item["content"])

        elif item_type == "audio":
            audio_index = len(audios) + 1
            role_content += f"<|audio_{audio_index}|>"
            audios.append(item["content"])

    # Add final role content
    if current_role is not None:
        role_token = user_token if current_role == "user" else assistant_token
        complete_prompt += f"{role_token}{role_content}{end_token}"

    # Add assistant token to prompt response
    if current_role != "assistant":
        complete_prompt += f"{assistant_token}"

    return complete_prompt, images, audios


def generate_response(
    model: AutoModelForCausalLM,
    processor: AutoProcessor,
    system_prompt: str,
    content_list: List[Dict],
    generation_config: Optional[GenerationConfig] = None,
    max_new_tokens: int = 1024,
    device: Optional[str] = None
) -> Tuple[str, str, Dict[str, float]]:
    """
    Generate a response from the model given content inputs.

    Args:
        model: Loaded Phi-4 model
        processor: Loaded processor
        system_prompt: System message defining behavior
        content_list: List of content items (text, image, audio)
        generation_config: Generation configuration
        max_new_tokens: Maximum tokens to generate
        device: Target device

    Returns:
        Tuple of (complete_prompt, response_text, timing_info)
    """
    device = device or get_device()
    timing_info = {}

    # Build the prompt
    complete_prompt, images, audios = build_prompt(system_prompt, content_list)

    # Process inputs
    with Timer() as t_proc:
        inputs = processor(
            text=complete_prompt,
            images=images if images else None,
            audios=audios if audios else None,
            return_tensors="pt"
        )

        # Move to device
        if device == "cuda":
            inputs = inputs.to("cuda:0")
        else:
            inputs = inputs.to("cpu")

    timing_info["processor_time"] = t_proc.elapsed

    # Generate response
    with Timer() as t_gen:
        if generation_config:
            generate_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                generation_config=generation_config,
            )
        else:
            generate_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
            )

    timing_info["generation_time"] = t_gen.elapsed

    # Extract only new tokens
    generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]

    # Decode response
    with Timer() as t_dec:
        response = processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

    timing_info["decode_time"] = t_dec.elapsed
    timing_info["total_time"] = (
        timing_info["processor_time"] +
        timing_info["generation_time"] +
        timing_info["decode_time"]
    )

    return complete_prompt, response, timing_info
