"""
Gozumol Prompts Module

This module contains all prompt templates used for visual assistance.
Prompts are carefully crafted to provide safe, actionable, and friendly
guidance for visually impaired users.
"""

from .templates import (
    # System prompts
    DEFAULT_SYSTEM_PROMPT,
    OUTDOOR_NAVIGATION_SYSTEM_PROMPT,
    INDOOR_NAVIGATION_SYSTEM_PROMPT,
    TRAFFIC_SAFETY_SYSTEM_PROMPT,
    CROWD_NAVIGATION_SYSTEM_PROMPT,

    # User prompts
    DEFAULT_USER_PROMPT,
    QUICK_SCAN_USER_PROMPT,
    DETAILED_DESCRIPTION_USER_PROMPT,
    SAFETY_CHECK_USER_PROMPT,
    CROSSING_ASSISTANCE_USER_PROMPT,

    # Prompt builders
    build_navigation_prompt,
    get_prompt_pair,
)

__all__ = [
    # System prompts
    "DEFAULT_SYSTEM_PROMPT",
    "OUTDOOR_NAVIGATION_SYSTEM_PROMPT",
    "INDOOR_NAVIGATION_SYSTEM_PROMPT",
    "TRAFFIC_SAFETY_SYSTEM_PROMPT",
    "CROWD_NAVIGATION_SYSTEM_PROMPT",

    # User prompts
    "DEFAULT_USER_PROMPT",
    "QUICK_SCAN_USER_PROMPT",
    "DETAILED_DESCRIPTION_USER_PROMPT",
    "SAFETY_CHECK_USER_PROMPT",
    "CROSSING_ASSISTANCE_USER_PROMPT",

    # Prompt builders
    "build_navigation_prompt",
    "get_prompt_pair",
]
