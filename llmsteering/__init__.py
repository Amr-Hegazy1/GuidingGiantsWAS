"""
LLM Safety Steering: A framework for steering large language models toward safer outputs.

This package implements a controller-based approach to dynamically modify model
activations during inference, encouraging models to refuse harmful prompts.
"""

import logging

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Import main components for easy access
from .controller.network import SteeringController, SteeringModifier
from .controller.training import train_controller, prepare_controller_inputs

# Version information
__version__ = "0.1.0"
