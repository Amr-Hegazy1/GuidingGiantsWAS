"""
Evaluation utilities for LLM Safety Steering.

This package provides evaluation utilities to assess model performance,
including refusal rate calculation using an LLM-based detector.
"""

from .refusal_rate import (
    RefusalDetector,
    calculate_batch_refusal_rate,
    evaluate_benchmark_results,
    print_evaluation_summary
)
