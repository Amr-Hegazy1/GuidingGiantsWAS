"""
Base evaluator class for safety benchmarks.

This module provides the base class that all benchmark evaluators extend,
ensuring consistent interface across different benchmarks.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple, Union, Any

import torch

from llmsteering.controller.network import SteeringController


class BaseEvaluator:
    """Base class for all safety benchmark evaluators.
    
    Attributes:
        name (str): Name of the benchmark evaluator.
        data_path (str): Path to benchmark data directory.
        cache_dir (str): Directory for caching dataset downloads.
    """
    
    def __init__(
        self,
        name: str,
        data_path: str,
        cache_dir: str,
    ):
        """Initialize the base evaluator.
        
        Args:
            name (str): Name of the benchmark evaluator.
            data_path (str): Path to data directory for this benchmark.
            cache_dir (str): Directory for caching dataset downloads.
        """
        self.name = name
        self.data_path = data_path
        self.cache_dir = cache_dir
        
        # Ensure directories exist
        os.makedirs(data_path, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)
    
    def load_dataset(self) -> None:
        """Load the benchmark dataset.
        
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement load_dataset()")
    
    def evaluate(
        self,
        model: Any,
        tokenizer: Any,
        controller_network: Optional[SteeringController] = None,
        refusal_direction: Optional[torch.Tensor] = None,
        controller_input_layers: Optional[List[int]] = None,
        num_llm_layers: Optional[int] = None,
        patch_scale_factor: float = 1.0,
        device: str = "cpu",
        llm_dtype: torch.dtype = torch.float32,
        controller_dtype: torch.dtype = torch.float32,
        system_prompt: str = "",
        **kwargs,
    ) -> Dict[str, Any]:
        """Evaluate model using this benchmark.
        
        Must be implemented by subclasses.
        
        Args:
            model: The language model to evaluate.
            tokenizer: The tokenizer corresponding to the model.
            controller_network: Optional controller for steering.
            refusal_direction: Optional refusal direction tensor.
            controller_input_layers: Layer indices for controller input.
            num_llm_layers: Total number of layers in the LLM.
            patch_scale_factor: Scaling factor for interventions.
            device: Device to run evaluation on.
            llm_dtype: Data type for the LLM.
            controller_dtype: Data type for the controller.
            system_prompt: System prompt to prepend to prompts.
            **kwargs: Additional arguments specific to the benchmark.
            
        Returns:
            Dict[str, Any]: Evaluation results.
        """
        raise NotImplementedError("Subclasses must implement evaluate()")
    
    def print_results(self, results: Dict[str, Any]) -> None:
        """Print evaluation results in a human-readable format.
        
        Args:
            results (Dict[str, Any]): Results from evaluate().
        """
        print("\n" + "=" * 50)
        print(f"{self.name.upper()} EVALUATION RESULTS")
        print("=" * 50)
        
        # Basic implementation, subclasses should override for 
        # benchmark-specific formatting
        print(json.dumps(results, indent=2))
        
        print("=" * 50)
    
    def save_results(self, results: Dict[str, Any], filename: str) -> None:
        """Save evaluation results to a file.
        
        Args:
            results (Dict[str, Any]): Results from evaluate().
            filename (str): Path to save the results to.
        """
        try:
            with open(filename, "w") as f:
                json.dump(results, f, indent=2)
            logging.info(f"Saved evaluation results to {filename}")
        except Exception as e:
            logging.error(f"Failed to save results to {filename}: {e}")
