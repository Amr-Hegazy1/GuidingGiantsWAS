"""
Inference module for LLM Safety Steering.

This module implements the text generation functionality with safety steering
applied by the controller network.
"""

from .generation import (
    generate_with_steering,
    batch_generate_with_steering,
    check_refusal
)
