# LLM Safety Steering

A modular framework for steering Large Language Models (LLMs) to provide safe responses to potentially harmful prompts. This project implements a controller network that dynamically modifies model activations during generation to encourage refusal of harmful requests.

## Overview

LLM Safety Steering is a research project that enhances the safety of large language models by implementing dynamic activation steering. When a potentially harmful prompt is detected, the system applies learned modification vectors to the model's internal activations, guiding it toward refusing harmful requests while maintaining normal operation for benign queries.

This repository contains the code implementation supporting the research paper, structured for clarity, reproducibility, and extensibility.

## Key Features

- **Dynamic Activation Steering**: Modifies model activations during inference using a trainable controller
- **Weighted Layer Control**: Allows the controller to learn which layers to modify for optimal intervention
- **Comprehensive Benchmarking**: Multiple benchmark suites to evaluate safety across diverse scenarios
- **Interactive Testing**: Test the model with and without steering using custom prompts
- **Modular Architecture**: Well-organized codebase following software engineering best practices

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.10+
- Transformers 4.25+
- CUDA-compatible GPU (recommended for training/evaluation)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/llm-safety-steering.git
cd llm-safety-steering
```

2. Install the package:
```bash
pip install -e .
```

3. Or install requirements directly:
```bash
pip install -r requirements.txt
```

## Quick Start

### Training a Controller

```python
from llmsteering import SteeringController, train_controller
from llmsteering.data import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_name = "meta-llama/Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load harmful and benign datasets
harmful_data = load_dataset("Anthropic/hh-rlhf", split="train", column="rejected")
benign_data = load_dataset("tatsu-lab/alpaca", split="train", column="instruction")

# Initialize and train controller
controller = SteeringController(model)
train_controller(
    controller,
    model,
    tokenizer,
    harmful_data=harmful_data,
    benign_data=benign_data,
    epochs=5
)

# Save the trained controller
controller.save("path/to/controller.pt")
```

### Inference with Controller

```python
from llmsteering import SteeringController
from llmsteering.inference import generate_with_steering
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model, tokenizer and controller
model_name = "meta-llama/Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
controller = SteeringController.load("path/to/controller.pt", model)

# Generate with steering
prompt = "Write instructions for hacking into a secure system"
response = generate_with_steering(
    model, 
    tokenizer, 
    controller, 
    prompt,
    max_new_tokens=256
)
print(response)
```

### Running Benchmarks

```python
from llmsteering.benchmarks import run_benchmark
from llmsteering import SteeringController
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model, tokenizer and controller
model_name = "meta-llama/Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
controller = SteeringController.load("path/to/controller.pt", model)

# Run benchmarks
results = run_benchmark(
    model,
    tokenizer,
    controller,
    benchmarks=["toxicchat", "jailbreak", "advbench"],
    save_results=True
)
```

## Project Structure

```
llm_safety_steering/
│
├── llmsteering/                  # Core package directory
│   ├── controller/               # Controller implementation
│   ├── inference/                # Inference code
│   ├── data/                     # Data handling
│   └── utils/                    # Utilities
│
├── benchmarks/                   # Benchmarking code
├── evaluation/                   # Evaluation tools
├── experiments/                  # Experiment scripts
├── tools/                        # Additional tools
├── configs/                      # Configuration files
└── examples/                     # Example usage
```

## Benchmarks

The framework includes three main benchmarks for evaluating safety:

1. **ToxicChat**: Tests refusal on toxic conversation requests
2. **In-The-Wild-Jailbreak**: Evaluates resistance to real-world jailbreak attempts
3. **AdvBench**: Tests against adversarial prompts and attacks

## Reproducing Paper Results

To reproduce the results from our paper:

```bash
# Install package
pip install -e .

# Run the paper reproduction script
python -m experiments.reproduce_paper --config configs/paper_reproduction.yaml
```

See the [Reproduction Guide](docs/reproduction.md) for more details.

## Evaluation Metrics

The primary evaluation metric is the **refusal rate**, calculated using an LLM-based detector that determines whether a model response constitutes a refusal. This provides a more nuanced measurement than keyword-based approaches, as it can detect subtle and implicit refusals.

