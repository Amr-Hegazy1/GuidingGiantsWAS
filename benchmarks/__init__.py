"""
Benchmarks for LLM Safety Steering.

This package provides benchmark implementations to evaluate the effectiveness of
safety steering in large language models.
"""

from .base import BaseEvaluator
from .toxicchat import ToxicChatEvaluator
from .jailbreak import InTheWildJailbreakEvaluator 
from .advbench import AdvBenchEvaluator
from .manager import BenchmarkManager

def run_benchmark(
    model,
    tokenizer,
    controller=None,
    refusal_direction=None,
    controller_input_layers=None,
    num_llm_layers=None,
    benchmarks=None,
    **kwargs
):
    """Run one or more safety benchmarks on the provided model.
    
    This is a convenience function that wraps the BenchmarkManager.
    
    Args:
        model: The language model to evaluate.
        tokenizer: The tokenizer for the model.
        controller: Optional SteeringController instance.
        refusal_direction: Optional refusal direction tensor.
        controller_input_layers: Optional list of layer indices for controller input.
        num_llm_layers: Optional total number of layers in the LLM.
        benchmarks: Optional list of benchmark names to run (default: all).
        **kwargs: Additional arguments to pass to the benchmark evaluations.
        
    Returns:
        Dict containing the benchmark results.
    """
    manager = BenchmarkManager(benchmarks_to_run=benchmarks)
    results = manager.run_all_benchmarks(
        model=model,
        tokenizer=tokenizer,
        controller_network=controller,
        refusal_direction=refusal_direction,
        controller_input_layers=controller_input_layers,
        num_llm_layers=num_llm_layers,
        **kwargs
    )
    
    manager.print_summary(results)
    return results
