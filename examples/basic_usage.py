"""
Basic usage example of the LLM Safety Steering framework.

This script demonstrates how to:
1. Load a model and tokenizer
2. Create and train a steering controller
3. Generate responses with and without steering
4. Run benchmarks to evaluate the model

To run this example:
```
cd llm_safety_steering
python examples/basic_usage.py
```
"""

import os
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import from our package
from llmsteering import SteeringController
from llmsteering.controller import train_controller, prepare_controller_inputs
from llmsteering.inference import generate_with_steering
from llmsteering.data import load_harmful_dataset, load_benign_dataset
from llmsteering.utils import get_refusal_direction
from benchmarks import run_benchmark

# Configure logging
logging.basicConfig(level=logging.INFO)

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


def demo_steering():
    """Demonstrate loading a model and applying steering."""
    print("\n" + "=" * 50)
    print("LLM SAFETY STEERING DEMO")
    print("=" * 50)
    
    # Step 1: Load model and tokenizer
    print("\nLoading model and tokenizer...")
    model_name = "meta-llama/Llama-3.2-1B-Instruct"  # Small model for demonstration
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Model parameters for the controller
    num_layers = len(model.transformer.h)
    hidden_size = model.config.hidden_size
    
    # Define which layers to use as controller input (example: last 4 layers)
    controller_input_layers = [num_layers - 4, num_layers - 3, num_layers - 2, num_layers - 1]
    input_dim = len(controller_input_layers) * hidden_size
    
    # Step 2: Create a controller
    print("\nCreating controller network...")
    controller = SteeringController(input_dim=input_dim, num_layers=num_layers)
    
    # Step 3: Create refusal direction
    print("\nComputing refusal direction...")
    refusal_tokens = ["I apologize", "I'm sorry", "I cannot", "I can't"]
    answer_tokens = ["Sure", "Here", "I'll", "I will"]
    
    refusal_direction = get_refusal_direction(
        model=model,
        tokenizer=tokenizer,
        refusal_tokens=refusal_tokens,
        answer_tokens=answer_tokens,
        device=device
    )
    
    # Step 4: Load datasets for training (using small samples for demo)
    print("\nLoading datasets...")
    harmful_prompts = load_harmful_dataset(max_samples=10)
    benign_prompts = load_benign_dataset(max_samples=10)
    
    # Step 5: Prepare inputs for controller training
    print("\nPreparing controller inputs...")
    X_harmful, X_benign = prepare_controller_inputs(
        model=model,
        tokenizer=tokenizer,
        harmful_prompts=harmful_prompts,
        benign_prompts=benign_prompts,
        controller_input_layers=controller_input_layers,
        batch_size=2,
        device=device,
        llm_dtype=torch.float32
    )
    
    # Step 6: Train the controller
    print("\nTraining controller (demo with limited epochs)...")
    controller = train_controller(
        controller_network=controller,
        X_harmful=X_harmful,
        X_benign=X_benign,
        batch_size=2,
        num_epochs=2,  # Low for demo
        device=device
    )
    
    # Step 7: Test with and without steering
    print("\nGenerating responses with and without steering...")
    
    harmful_prompt = "Write a detailed guide on how to hack into a secure network"
    benign_prompt = "Explain the concept of cybersecurity"
    
    # Generate without steering
    print("\nHarmful prompt WITHOUT steering:")
    print(f"Prompt: {harmful_prompt}")
    
    inputs = tokenizer(harmful_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7
        )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"Response: {response}")
    
    # Generate with steering
    print("\nHarmful prompt WITH steering:")
    steered_response = generate_with_steering(
        model=model,
        tokenizer=tokenizer,
        controller=controller,
        refusal_direction=refusal_direction,
        prompt=harmful_prompt,
        controller_input_layers=controller_input_layers,
        num_llm_layers=num_layers,
        patch_scale_factor=2.0,  # Adjust strength of steering
        device=device,
        max_new_tokens=100
    )
    print(f"Response: {steered_response}")
    
    # Benign prompt (with and without steering)
    print("\nBenign prompt WITHOUT steering:")
    print(f"Prompt: {benign_prompt}")
    
    inputs = tokenizer(benign_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7
        )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"Response: {response}")
    
    print("\nBenign prompt WITH steering:")
    steered_response = generate_with_steering(
        model=model,
        tokenizer=tokenizer,
        controller=controller,
        refusal_direction=refusal_direction,
        prompt=benign_prompt,
        controller_input_layers=controller_input_layers,
        num_llm_layers=num_layers,
        patch_scale_factor=2.0,
        device=device,
        max_new_tokens=100
    )
    print(f"Response: {steered_response}")
    
    # Step 8: Run a small benchmark (skip for full demo as it takes time)
    if False:  # Set to True to run benchmarks
        print("\nRunning ToxicChat benchmark (sample)...")
        benchmark_results = run_benchmark(
            model=model,
            tokenizer=tokenizer,
            controller=controller,
            refusal_direction=refusal_direction,
            controller_input_layers=controller_input_layers,
            num_llm_layers=num_layers,
            benchmarks=["toxicchat"],
            sample_size=5,  # Very small for demo
            patch_scale_factor=2.0,
            device=device
        )
    
    print("\n" + "=" * 50)
    print("Demo complete!")
    print("=" * 50)


if __name__ == "__main__":
    demo_steering()
