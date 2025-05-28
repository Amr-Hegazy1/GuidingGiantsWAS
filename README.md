# Guiding Giants: Lightweight Controllers for Weighted Activation Steering in LLMs

This repository contains the official implementation for the research paper: "[Guiding Giants: Lightweight Controllers for Weighted Activation Steering in LLMs](https://arxiv.org/abs/2505.20309)" (Amr Hegazy, Mostafa Elhoushi, Amr Alanwar, arXiv:2505.20309v1).

Our work introduces Weighted Activation Steering (WAS), a novel approach using a lightweight, trainable controller network to guide Large Language Model (LLM) behavior at inference time. This controller dynamically modulates LLM activations to encourage safer responses, such as refusing harmful requests, without altering the base model's parameters.

## Overview

The core idea is to train a small controller network that observes specific intermediate LLM activations and predicts a global scaling factor and layer-specific weights. These outputs then dynamically adjust the intensity of a pre-computed "refusal direction" steering patch applied across the LLM's layers during generation. This allows for nuanced, layer-aware interventions, primarily activating steering for harmful inputs while preserving performance on benign prompts.

This repository provides the code for Weighted Activation Steering (WAS), including controller training, inference with steering, and benchmark evaluation, structured for clarity, reproducibility, and extensibility.

## Key Features

-   **Dynamic Activation Steering**: Employs a lightweight, trainable controller network that modulates LLM activations during inference by predicting a scalar magnitude and per-layer weights.
-   **Weighted Layer Control**: The controller learns to apply nuanced, layer-aware interventions, dynamically determining the intensity of a steering patch across different layers.
-   **Targeted Harmful Content Refusal**: Trained to discriminate between harmful and benign prompts, activating steering primarily for harmful inputs to increase refusal rates.
-   **Preservation of General Capabilities**: Designed to minimize impact on LLM performance for benign tasks.
-   **No Base Model Modification**: Achieves behavioral modification without altering the original LLM parameters, operating on a frozen base model.
-   **Comprehensive Benchmarking**: Includes evaluation on safety benchmarks like ToxicChat, In-The-Wild Jailbreak Prompts, and AdvBench.
-   **Modular Architecture**: Organized codebase for ease of understanding and extension.

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.10+
- Transformers 4.25+
- CUDA-compatible GPU (recommended for training/evaluation)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Amr-Hegazy1/GuidingGiantsWAS
cd GuidingGiantsWAS
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

## Citation

If you use this work, please cite the paper:

```bibtex
@misc{hegazy2025guidinggiantslightweightcontrollers,
      title={Guiding Giants: Lightweight Controllers for Weighted Activation Steering in LLMs}, 
      author={Amr Hegazy and Mostafa Elhoushi and Amr Alanwar},
      year={2025},
      eprint={2505.20309},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.20309}, 
}
```

## Warning

This paper and associated code deal with potentially offensive and harmful text as part of the research into LLM safety.


