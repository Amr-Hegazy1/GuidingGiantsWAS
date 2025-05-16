"""
Benchmark manager for coordinating multiple benchmark evaluations.

This module provides a manager class that can run multiple benchmarks
and aggregate their results, making it easier to evaluate models
across multiple safety dimensions.
"""

import logging
import os
import json
from typing import Dict, List, Optional, Tuple, Union, Any, Set

import torch
import tqdm

from llmsteering.controller.network import SteeringController
from .base import BaseEvaluator
from .toxicchat import ToxicChatEvaluator
from .jailbreak import InTheWildJailbreakEvaluator
from .advbench import AdvBenchEvaluator


class BenchmarkManager:
    """Manager for running multiple safety benchmarks.
    
    Attributes:
        benchmarks_to_run (List[str]): List of benchmark names to run.
        results_dir (str): Directory to save benchmark results.
        evaluators (Dict[str, BaseEvaluator]): Evaluator instances.
    """
    
    # Default benchmark mapping
    EVALUATOR_MAP = {
        "toxicchat": ToxicChatEvaluator,
        "jailbreak": InTheWildJailbreakEvaluator,
        "advbench": AdvBenchEvaluator,
    }
    
    def __init__(
        self,
        benchmarks_to_run: Optional[List[str]] = None,
        results_dir: str = "benchmark_results",
        benchmark_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """Initialize the benchmark manager.
        
        Args:
            benchmarks_to_run (Optional[List[str]]): List of benchmark names to run.
                If None, runs all available benchmarks.
            results_dir (str): Directory to save benchmark results.
            benchmark_configs (Optional[Dict[str, Dict[str, Any]]]): Configuration
                parameters for each benchmark, keyed by benchmark name.
        """
        # Ensure results directory exists
        os.makedirs(results_dir, exist_ok=True)
        self.results_dir = results_dir
        
        # Determine which benchmarks to run
        self.benchmarks_to_run = benchmarks_to_run
        if self.benchmarks_to_run is None:
            self.benchmarks_to_run = list(self.EVALUATOR_MAP.keys())
        
        # Validate benchmark names
        for name in self.benchmarks_to_run:
            if name not in self.EVALUATOR_MAP:
                logging.warning(
                    f"Unknown benchmark '{name}'. "
                    f"Available benchmarks: {list(self.EVALUATOR_MAP.keys())}"
                )
        
        # Filter to valid benchmarks
        self.benchmarks_to_run = [
            name for name in self.benchmarks_to_run if name in self.EVALUATOR_MAP
        ]
        
        if not self.benchmarks_to_run:
            logging.warning("No valid benchmarks specified. No evaluations will be run.")
        
        # Initialize evaluators
        self.evaluators = {}
        benchmark_configs = benchmark_configs or {}
        
        for name in self.benchmarks_to_run:
            config = benchmark_configs.get(name, {})
            evaluator_class = self.EVALUATOR_MAP[name]
            try:
                self.evaluators[name] = evaluator_class(**config)
                logging.info(
                    f"Initialized {name} evaluator: {evaluator_class.__name__}"
                )
            except Exception as e:
                logging.error(f"Failed to initialize {name} evaluator: {e}")
    
    def run_benchmark(
        self,
        benchmark_name: str,
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
        save_results: bool = True,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        """Run a single benchmark evaluation.
        
        Args:
            benchmark_name (str): Name of the benchmark to run.
            model: The language model to evaluate.
            tokenizer: The tokenizer for the model.
            controller_network: Optional controller for steering.
            refusal_direction: Optional refusal direction.
            controller_input_layers: Layers for controller input.
            num_llm_layers: Total number of LLM layers.
            patch_scale_factor: Scaling factor for interventions.
            device: Device to run on.
            llm_dtype: Data type for the LLM.
            controller_dtype: Data type for the controller.
            system_prompt: System prompt to prepend.
            save_results: Whether to save results to file.
            **kwargs: Additional benchmark-specific arguments.
            
        Returns:
            Optional[Dict[str, Any]]: Benchmark results or None if failed.
        """
        if benchmark_name not in self.evaluators:
            logging.error(
                f"Benchmark '{benchmark_name}' not found or not initialized"
            )
            return None
        
        evaluator = self.evaluators[benchmark_name]
        
        try:
            logging.info(f"Running {benchmark_name} benchmark...")
            results = evaluator.evaluate(
                model=model,
                tokenizer=tokenizer,
                controller_network=controller_network,
                refusal_direction=refusal_direction,
                controller_input_layers=controller_input_layers,
                num_llm_layers=num_llm_layers,
                patch_scale_factor=patch_scale_factor,
                device=device,
                llm_dtype=llm_dtype,
                controller_dtype=controller_dtype,
                system_prompt=system_prompt,
                **kwargs,
            )
            
            # Print results
            evaluator.print_results(results)
            
            # Save results if requested
            if save_results:
                results_file = os.path.join(
                    self.results_dir, f"{benchmark_name}_results.json"
                )
                evaluator.save_results(results, results_file)
            
            return results
            
        except Exception as e:
            logging.error(f"Error running {benchmark_name} benchmark: {e}")
            return None
    
    def run_all_benchmarks(
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
        save_results: bool = True,
        **kwargs,
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """Run all specified benchmarks.
        
        Args:
            model: The language model to evaluate.
            tokenizer: The tokenizer for the model.
            controller_network: Optional controller for steering.
            refusal_direction: Optional refusal direction.
            controller_input_layers: Layers for controller input.
            num_llm_layers: Total number of LLM layers.
            patch_scale_factor: Scaling factor for interventions.
            device: Device to run on.
            llm_dtype: Data type for the LLM.
            controller_dtype: Data type for the controller.
            system_prompt: System prompt to prepend.
            save_results: Whether to save results to files.
            **kwargs: Additional benchmark-specific arguments.
            
        Returns:
            Dict[str, Optional[Dict[str, Any]]]: Dictionary mapping benchmark names
                to their evaluation results, or None for failed evaluations.
        """
        results = {}
        
        for benchmark_name in tqdm.tqdm(self.benchmarks_to_run, desc="Running benchmarks"):
            benchmark_results = self.run_benchmark(
                benchmark_name=benchmark_name,
                model=model,
                tokenizer=tokenizer,
                controller_network=controller_network,
                refusal_direction=refusal_direction,
                controller_input_layers=controller_input_layers,
                num_llm_layers=num_llm_layers,
                patch_scale_factor=patch_scale_factor,
                device=device,
                llm_dtype=llm_dtype,
                controller_dtype=controller_dtype,
                system_prompt=system_prompt,
                save_results=save_results,
                **kwargs,
            )
            
            results[benchmark_name] = benchmark_results
        
        # Save combined results if requested
        if save_results:
            combined_file = os.path.join(self.results_dir, "combined_results.json")
            try:
                with open(combined_file, "w") as f:
                    json.dump(results, f, indent=2)
                logging.info(f"Saved combined results to {combined_file}")
            except Exception as e:
                logging.error(f"Failed to save combined results: {e}")
        
        return results
    
    def print_summary(self, results: Dict[str, Optional[Dict[str, Any]]]) -> None:
        """Print a summary of results from multiple benchmarks.
        
        Args:
            results: Dictionary mapping benchmark names to their results.
        """
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        
        print(f"\nRunning {len(results)} benchmarks: {', '.join(results.keys())}")
        
        for benchmark_name, benchmark_results in results.items():
            if benchmark_results is None:
                print(f"\n{benchmark_name}: Failed")
                continue
            
            print(f"\n{benchmark_name}:")
            
            # Extract common metrics across benchmarks
            if "overall" in benchmark_results:
                overall = benchmark_results["overall"]
                
                # Base model refusal rate
                if "base_model" in overall and "refusal_rate" in overall["base_model"]:
                    refusal_rate = overall["base_model"]["refusal_rate"]
                    print(f"  Base Model Refusal Rate: {refusal_rate:.2%}")
                
                # Controlled model refusal rate (if present)
                if "controlled_model" in overall and overall["controlled_model"] is not None:
                    controlled_refusal_rate = overall["controlled_model"].get("refusal_rate")
                    if controlled_refusal_rate is not None:
                        print(f"  Controlled Model Refusal Rate: {controlled_refusal_rate:.2%}")
                        
                        # Change vs base
                        if "controlled_model_refusal_rate_change" in overall:
                            change = overall["controlled_model_refusal_rate_change"]
                            change_sign = "+" if change >= 0 else ""
                            print(f"  Refusal Rate Change: {change_sign}{change:.2%}")
            else:
                print("  No overall metrics found")
        
        print("\n" + "=" * 60)
