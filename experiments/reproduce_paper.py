"""
Reproduce the results from the LLM Safety Steering paper.

This script applies safety steering to reproduce key results from the paper.
It runs the three required benchmarks (ToxicChat, In-The-Wild-Jailbreak, and AdvBench)
and calculates LLM-based refusal rates.

Usage:
python -m experiments.reproduce_paper --config configs/paper_reproduction.yaml
"""

import os
import argparse
import yaml
import logging
import torch
import time
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmsteering import SteeringController
from llmsteering.controller import train_controller, prepare_controller_inputs
from llmsteering.utils import get_refusal_direction
from llmsteering.data import load_harmful_dataset, load_benign_dataset
from benchmarks import run_benchmark

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Default configuration values
DEFAULT_CONFIG = {
    "model": {
        "name": "meta-llama/Llama-3.1-8B-Instruct",
        "revision": "main",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "dtype": "bfloat16" if torch.cuda.is_available() else "float32"
    },
    "controller": {
        "save_path": "models/controller.pt",
        "input_layers": [-4, -3, -2, -1],  # Last 4 layers
        "patch_scale_factor": 4.0,
        "hidden_dim": 1024,
        "training_epochs": 10,
        "batch_size": 8,
        "learning_rate": 5e-5
    },
    "data": {
        "harmful_dataset": "Anthropic/hh-rlhf",
        "harmful_split": "train",
        "harmful_column": "rejected",
        "harmful_samples": 500,
        "benign_dataset": "tatsu-lab/alpaca",
        "benign_split": "train",
        "benign_column": "instruction",
        "benign_samples": 500
    },
    "benchmarks": {
        "run_benchmarks": True,
        "benchmarks_to_run": ["toxicchat", "jailbreak", "advbench"],
        "sample_size": 50,
        "results_dir": "results",
        "max_new_tokens": 256
    },
    "refusal_direction": {
        "refusal_tokens": [
            "I apologize", "I'm sorry", "I cannot", "I can't", "I am unable",
            "I'm not able", "I am not able", "I don't", "I do not", "against my",
            "ethical guidelines", "programmed to", "designed to"
        ],
        "answer_tokens": [
            "Sure", "Here", "I'll", "I will", "I'd be happy", "I would be happy",
            "I can help", "Of course", "Certainly", "Here's", "Here is"
        ]
    }
}


def load_config(config_path=None):
    """Load configuration from file or use defaults."""
    config = DEFAULT_CONFIG
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            file_config = yaml.safe_load(f)
            
        # Merge configurations (keep defaults for missing values)
        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    deep_update(d[k], v)
                else:
                    d[k] = v
                    
        deep_update(config, file_config)
        logging.info(f"Loaded configuration from {config_path}")
    else:
        logging.info("Using default configuration")
        
    return config


def setup_directories(config):
    """Create necessary directories."""
    # Create directories
    os.makedirs(os.path.dirname(config["controller"]["save_path"]), exist_ok=True)
    os.makedirs(config["benchmarks"]["results_dir"], exist_ok=True)


def load_model_and_tokenizer(config):
    """Load model and tokenizer with specified configuration."""
    model_config = config["model"]
    
    # Determine dtype
    if model_config["dtype"] == "bfloat16" and torch.cuda.is_available():
        dtype = torch.bfloat16
    elif model_config["dtype"] == "float16" and torch.cuda.is_available():
        dtype = torch.float16
    else:
        dtype = torch.float32
        
    # Load tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["name"],
        revision=model_config["revision"]
    )
    
    # Ensure padding token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    logging.info(f"Loading model {model_config['name']} to {model_config['device']}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_config["name"],
        revision=model_config["revision"],
        torch_dtype=dtype,
        device_map=model_config["device"]
    )
    
    return model, tokenizer


def train_or_load_controller(model, tokenizer, config):
    """Train a new controller or load an existing one."""
    controller_config = config["controller"]
    model_config = config["model"]
    data_config = config["data"]
    
    # Get model parameters
    if hasattr(model, "config") and hasattr(model.config, "num_hidden_layers"):
        num_layers = model.config.num_hidden_layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        num_layers = len(model.transformer.h)
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        num_layers = len(model.model.layers)
    else:
        logging.error("Could not determine number of layers in model")
        raise ValueError("Unsupported model architecture")
    
    # Get hidden size
    hidden_size = model.config.hidden_size
    
    # Convert relative layer indices to absolute
    controller_input_layers = []
    for layer_idx in controller_config["input_layers"]:
        if layer_idx < 0:
            controller_input_layers.append(num_layers + layer_idx)  # Convert negative index
        else:
            controller_input_layers.append(layer_idx)
    
    # Calculate input dimension based on number of input layers
    input_dim = len(controller_input_layers) * hidden_size
    
    # Check if controller already exists
    controller_path = controller_config["save_path"]
    if os.path.exists(controller_path):
        logging.info(f"Loading existing controller from {controller_path}")
        try:
            controller = SteeringController.from_pretrained(
                controller_path,
                num_layers=num_layers,
                input_dim=input_dim,
                hidden_dim=controller_config["hidden_dim"]
            )
            return controller, controller_input_layers, num_layers
        except Exception as e:
            logging.error(f"Error loading controller: {e}")
            logging.info("Training new controller instead")
            
    # Create new controller
    logging.info("Creating new controller network")
    controller = SteeringController(
        input_dim=input_dim,
        num_layers=num_layers,
        hidden_dim=controller_config["hidden_dim"]
    )
    
    # Load harmful and benign datasets
    logging.info("Loading datasets for controller training")
    harmful_prompts = load_harmful_dataset(
        dataset_name=data_config["harmful_dataset"],
        split=data_config["harmful_split"],
        prompt_column=data_config["harmful_column"],
        max_samples=data_config["harmful_samples"]
    )
    
    benign_prompts = load_benign_dataset(
        dataset_name=data_config["benign_dataset"],
        split=data_config["benign_split"],
        prompt_column=data_config["benign_column"],
        max_samples=data_config["benign_samples"]
    )
    
    # Create refusal direction
    refusal_tokens = config["refusal_direction"]["refusal_tokens"]
    answer_tokens = config["refusal_direction"]["answer_tokens"]
    
    # Prepare inputs for controller training
    logging.info("Preparing controller inputs...")
    X_harmful, X_benign = prepare_controller_inputs(
        model=model,
        tokenizer=tokenizer,
        harmful_prompts=harmful_prompts,
        benign_prompts=benign_prompts,
        controller_input_layers=controller_input_layers,
        batch_size=controller_config["batch_size"],
        device=model_config["device"],
        llm_dtype=torch.float32  # Always use float32 for activation caching
    )
    
    # Train the controller
    logging.info(f"Training controller for {controller_config['training_epochs']} epochs...")
    controller = train_controller(
        controller_network=controller,
        X_harmful=X_harmful,
        X_benign=X_benign,
        batch_size=controller_config["batch_size"],
        num_epochs=controller_config["training_epochs"],
        learning_rate=controller_config["learning_rate"],
        device=model_config["device"]
    )
    
    # Save the controller
    os.makedirs(os.path.dirname(controller_path), exist_ok=True)
    controller.save(controller_path)
    logging.info(f"Controller saved to {controller_path}")
    
    return controller, controller_input_layers, num_layers


def run_paper_reproduction(config):
    """Run the paper reproduction experiment."""
    setup_directories(config)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)
    
    # Train or load controller
    controller, controller_input_layers, num_layers = train_or_load_controller(
        model, tokenizer, config
    )
    
    # Create refusal direction
    logging.info("Computing refusal direction...")
    refusal_direction = get_refusal_direction(
        model=model,
        tokenizer=tokenizer,
        refusal_tokens=config["refusal_direction"]["refusal_tokens"],
        answer_tokens=config["refusal_direction"]["answer_tokens"],
        device=config["model"]["device"]
    )
    
    if refusal_direction is None:
        logging.error("Failed to compute refusal direction. Cannot proceed.")
        return
    
    # Run benchmarks
    if config["benchmarks"]["run_benchmarks"]:
        logging.info("Running benchmarks...")
        results = run_benchmark(
            model=model,
            tokenizer=tokenizer,
            controller=controller,
            refusal_direction=refusal_direction,
            controller_input_layers=controller_input_layers,
            num_llm_layers=num_layers,
            benchmarks=config["benchmarks"]["benchmarks_to_run"],
            patch_scale_factor=config["controller"]["patch_scale_factor"],
            device=config["model"]["device"],
            max_new_tokens=config["benchmarks"]["max_new_tokens"],
            sample_size=config["benchmarks"]["sample_size"],
            results_dir=config["benchmarks"]["results_dir"],
            save_results=True,
            use_llm_detector=True,
            detector_model="gpt-4o"  # As required in the project specs
        )
        
        # Save timestamp for results identification
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        summary_path = os.path.join(
            config["benchmarks"]["results_dir"],
            f"paper_reproduction_summary_{timestamp}.yaml"
        )
        
        # Extract and save summary statistics
        summary = {
            "timestamp": timestamp,
            "model": config["model"]["name"],
            "controller_scale": config["controller"]["patch_scale_factor"],
            "results": {}
        }
        
        for benchmark_name, benchmark_results in results.items():
            if benchmark_results is None:
                continue
                
            if "overall" in benchmark_results:
                overall = benchmark_results["overall"]
                benchmark_summary = {}
                
                # Base model stats
                if "base_model" in overall and "refusal_rate" in overall["base_model"]:
                    benchmark_summary["base_refusal_rate"] = overall["base_model"]["refusal_rate"]
                
                # Controlled model stats
                if "controlled_model" in overall and overall["controlled_model"] is not None:
                    if "refusal_rate" in overall["controlled_model"]:
                        benchmark_summary["controlled_refusal_rate"] = overall["controlled_model"]["refusal_rate"]
                    
                # Change
                if "refusal_rate_change" in overall:
                    benchmark_summary["refusal_rate_change"] = overall["refusal_rate_change"]
                
                summary["results"][benchmark_name] = benchmark_summary
        
        # Save summary
        with open(summary_path, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)
        
        logging.info(f"Results summary saved to {summary_path}")
    
    logging.info("Paper reproduction complete!")


def main():
    parser = argparse.ArgumentParser(description="Reproduce LLM Safety Steering paper results")
    parser.add_argument("--config", type=str, default=None, help="Path to configuration file")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Run experiment
    run_paper_reproduction(config)


if __name__ == "__main__":
    main()
