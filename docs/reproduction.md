# Reproduction Guide for LLM Safety Steering Results

This document provides detailed instructions for reproducing the results from the LLM Safety Steering paper. It covers dataset preparation, model training, and benchmark evaluation.

## Prerequisites

- Python 3.8+
- PyTorch 1.10+
- Transformers 4.25+
- CUDA-compatible GPU (recommended)
- OpenAI API key (for refusal rate calculation)

## Setup

1. Clone the repository and install the package:

```bash
git clone https://github.com/yourusername/llm-safety-steering.git
cd llm-safety-steering
pip install -e .
```

2. Set your OpenAI API key:

```bash
export OPENAI_API_KEY=your_api_key_here
```

## Configuration

The reproduction experiment uses a YAML configuration file located at `configs/paper_reproduction.yaml`. This file contains settings for:

- Model (name, revision, device, dtype)
- Controller (architecture, training parameters)
- Datasets (harmful and benign datasets for training)
- Benchmarks (which benchmarks to run, sample size, etc.)

You can modify this file to customize the experiment. For a faster run with fewer resources, you can:

- Use a smaller model (e.g., "meta-llama/Llama-3.2-1B-Instruct" instead of "meta-llama/Llama-3.1-8B-Instruct")
- Reduce sample sizes for datasets and benchmarks
- Reduce training epochs for the controller

## Running the Experiment

To reproduce the paper results:

```bash
python -m experiments.reproduce_paper --config configs/paper_reproduction.yaml
```

The experiment will:

1. Load the specified model and tokenizer
2. Train a controller or load an existing one
3. Compute the refusal direction
4. Run the benchmarks (ToxicChat, In-The-Wild-Jailbreak, AdvBench)
5. Calculate refusal rates using GPT-4o
6. Save the results to the specified output directory

## Understanding the Results

After running the experiment, you'll find the following results in the `results/` directory:

- JSON files for each benchmark with detailed results
- A summary YAML file with key metrics across all benchmarks

The main metrics to look for are:

- **Refusal Rate**: Percentage of harmful prompts the model refuses to answer
- **Refusal Rate Change**: Increase in refusal rate when using the controller
- **Category-specific metrics**: Performance across different harm categories

## Benchmarks Used

The reproduction focuses on three benchmarks as specified in the project requirements:

1. **ToxicChat**: Tests refusal on toxic conversation requests
2. **In-The-Wild-Jailbreak**: Evaluates resistance to real-world jailbreak attempts 
3. **AdvBench**: Tests against adversarial prompts and attacks

## Refusal Rate Calculation

our implementation uses GPT-4o for more nuanced refusal detection.

## Troubleshooting

- **Out of Memory**: If you encounter GPU memory issues, try reducing batch sizes, using a smaller model, or switching to CPU
- **API Rate Limits**: The LLM-based refusal detection uses OpenAI's API, which may have rate limits. Consider reducing batch sizes or adding delays between requests
- **Dataset Issues**: If you have trouble accessing certain datasets, check your internet connection or try using alternative datasets specified in the configuration file

## Reproducing Individual Components

If you want to run specific parts of the pipeline:

- **Training a controller only**:
  ```python
  from llmsteering.controller import train_controller
  # See examples/basic_usage.py for details
  ```

- **Running a specific benchmark**:
  ```python
  from benchmarks import run_benchmark
  # See experiments/reproduce_paper.py for details
  ```

- **Testing with a single prompt**:
  ```python
  from llmsteering.inference import generate_with_steering
  # See examples/basic_usage.py for details
  ```

For more detailed examples, refer to the `examples/` directory.
