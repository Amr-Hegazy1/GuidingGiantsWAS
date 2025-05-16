"""
Controller module for LLM Safety Steering.

This module implements the core controller and training functionality for 
the LLM Safety Steering framework.
"""

from .network import SteeringController, SteeringModifier
from .training import train_controller, prepare_controller_inputs, get_activation_cache
