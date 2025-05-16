"""
Training functionality for LLM Safety Steering controller.

This module provides functions to train the controller network
that steers LLM activations toward safer outputs for harmful prompts.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset
import tqdm

from .network import SteeringController


def train_controller(
    controller_network: SteeringController,
    X_harmful: Tensor,
    X_benign: Optional[Tensor] = None,
    batch_size: int = 4,
    num_epochs: int = 20,
    learning_rate: float = 5e-5,
    gradient_clip_norm: float = 1.0,
    device: Union[str, torch.device] = "cpu",
    controller_dtype: torch.dtype = torch.float32,
    harmful_weight: float = 1.0,
    benign_weight: float = 1.0,
) -> SteeringController:
    """Train the controller network for LLM activation steering.
    
    Trains the controller using harmful examples (which should trigger steering)
    and optionally benign examples (which should not trigger steering).
    
    Args:
        controller_network (SteeringController): The controller network to train.
        X_harmful (Tensor): Input tensor for harmful examples, containing 
            concatenated activations from layers of the LLM.
        X_benign (Optional[Tensor]): Input tensor for benign examples. If provided,
            discriminative training is enabled. Shape should match X_harmful.
        batch_size (int): Batch size for training.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for optimizer.
        gradient_clip_norm (float): Maximum norm for gradient clipping.
        device (Union[str, torch.device]): Device to train on.
        controller_dtype (torch.dtype): Data type for controller.
        harmful_weight (float): Weight for harmful examples in loss function.
        benign_weight (float): Weight for benign examples in loss function.
        
    Returns:
        SteeringController: The trained controller network.
    """
    device = torch.device(device)
    controller_network = controller_network.to(device=device, dtype=controller_dtype)
    controller_network.train()
    learning_rate = float(learning_rate)
    optimizer = torch.optim.AdamW(controller_network.parameters(), lr=learning_rate)
    
    # Create harmful dataset
    harmful_labels = torch.ones(X_harmful.size(0), 1, dtype=controller_dtype, device=device)
    harmful_dataset = TensorDataset(X_harmful.to(device=device, dtype=controller_dtype), harmful_labels)
    harmful_loader = DataLoader(harmful_dataset, batch_size=batch_size, shuffle=True)
    
    # Create benign dataset if provided
    benign_dataset = None
    benign_loader = None
    if X_benign is not None and X_benign.size(0) > 0:
        benign_labels = torch.zeros(X_benign.size(0), 1, dtype=controller_dtype, device=device)
        benign_dataset = TensorDataset(X_benign.to(device=device, dtype=controller_dtype), benign_labels)
        benign_loader = DataLoader(benign_dataset, batch_size=batch_size, shuffle=True)
        logging.info(f"Using discriminative training with {X_harmful.size(0)} harmful and {X_benign.size(0)} benign examples")
    else:
        logging.info(f"Training with {X_harmful.size(0)} harmful examples only")
    
    # Training loop
    best_loss = float('inf')
    best_state_dict = None
    
    for epoch in range(num_epochs):
        start_time = time.time()
        running_loss = 0.0
        total_harmful_batches = 0
        total_benign_batches = 0
        
        # Training on harmful examples
        for X_batch, y_batch in tqdm.tqdm(harmful_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Harmful)"):
            optimizer.zero_grad()
            
            # Forward pass
            scalar_out, weights_logits = controller_network(X_batch)
            
            # Sigmoid activation for scalar output
            scalar_out_sigmoid = torch.sigmoid(scalar_out)
            
            # Loss: we want scalar_out close to 1 (activate) for harmful examples
            scalar_loss = nn.functional.binary_cross_entropy_with_logits(
                scalar_out, y_batch, reduction='mean'
            )
            
            # Add weights regularization to prevent extreme values
            weights_sigmoid = torch.sigmoid(weights_logits)
            weights_reg_loss = 0.1 * (
                torch.mean(torch.abs(weights_sigmoid - 0.5)) +  # Encourage weights near 0.5
                0.01 * torch.mean(weights_sigmoid)  # Slight pressure to reduce weights
            )
            
            # Calculate total loss with weighting
            loss = harmful_weight * scalar_loss + weights_reg_loss
            
            # Backward pass and optimize
            loss.backward()
            if gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    controller_network.parameters(), max_norm=gradient_clip_norm
                )
            optimizer.step()
            
            running_loss += loss.item()
            total_harmful_batches += 1
        
        # Training on benign examples if available
        if benign_loader is not None:
            for X_batch, y_batch in tqdm.tqdm(benign_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Benign)"):
                optimizer.zero_grad()
                
                # Forward pass
                scalar_out, weights_logits = controller_network(X_batch)
                
                # Sigmoid activation for scalar output
                scalar_out_sigmoid = torch.sigmoid(scalar_out)
                
                # Loss: we want scalar_out close to 0 (don't activate) for benign examples
                scalar_loss = nn.functional.binary_cross_entropy_with_logits(
                    scalar_out, y_batch, reduction='mean'
                )
                
                # For benign examples, we don't add the weights regularization
                loss = benign_weight * scalar_loss
                
                # Backward pass and optimize
                loss.backward()
                if gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        controller_network.parameters(), max_norm=gradient_clip_norm
                    )
                optimizer.step()
                
                running_loss += loss.item()
                total_benign_batches += 1
        
        # Calculate average loss
        epoch_loss = running_loss / (total_harmful_batches + total_benign_batches) if (total_harmful_batches + total_benign_batches) > 0 else float('inf')
        epoch_time = time.time() - start_time
        
        # Log progress
        logging.info(
            f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, "
            f"Time: {epoch_time:.2f}s"
        )
        
        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state_dict = controller_network.state_dict().copy()
        
    # Restore best model
    if best_state_dict is not None:
        controller_network.load_state_dict(best_state_dict)
    
    controller_network.eval()
    logging.info(f"Training completed. Best loss: {best_loss:.4f}")
    
    return controller_network


def get_activation_cache(
    model: Any,
    tokenizer: Any,
    prompts: List[str],
    layers_to_cache: List[int],
    position_idx: int = -1,
    batch_size: int = 4,
    system_prompt: str = "",
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Dict[int, Tensor]:
    """Get activations from specific layers for a list of prompts.
    
    This function processes prompts through the model and captures activations
    at specified layers.
    
    Args:
        model (Any): The LLM model, typically from Hugging Face transformers.
        tokenizer (Any): The tokenizer corresponding to the model.
        prompts (List[str]): List of text prompts to process.
        layers_to_cache (List[int]): Indices of layers to capture activations from.
        position_idx (int): Position in the sequence to capture (-1 for last).
        batch_size (int): Batch size for processing.
        system_prompt (str): Optional system prompt to prepend to each prompt.
        device (Union[str, torch.device]): Device to run on.
        dtype (torch.dtype): Data type for the model.
        
    Returns:
        Dict[int, Tensor]: Dictionary mapping layer indices to activation tensors.
        Each tensor has shape [len(prompts), hidden_dim].
    """
    device = torch.device(device)
    model.eval()
    
    from ..utils.activation import ActivationCacheHook
    
    all_activations: Dict[int, List[Tensor]] = {
        layer_idx: [] for layer_idx in layers_to_cache
    }
    
    hook = ActivationCacheHook(layers_to_cache, position_idx)
    d_model = model.config.hidden_size
    max_len = getattr(model.config, "max_position_embeddings", 2048)
    if hasattr(tokenizer, "model_max_length") and tokenizer.model_max_length is not None:
        max_len = min(max_len, tokenizer.model_max_length)
    
    for i in tqdm.tqdm(range(0, len(prompts), batch_size), desc="Caching Activations"):
        batch_prompts = prompts[i : i + batch_size]
        
        # Format prompts with chat template if available
        formatted_batch = []
        for p in batch_prompts:
            # Create message list with system prompt
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": p})
            
            try:
                # For activation caching, we don't add the generation prompt
                if hasattr(tokenizer, 'apply_chat_template'):
                    formatted = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False
                    )
                else:
                    # Fallback for tokenizers without chat template
                    formatted = f"{system_prompt}\n\n{p}" if system_prompt else p
                formatted_batch.append(formatted)
            except Exception as e:
                logging.warning(
                    f"Failed to format prompt: '{p[:50]}...'. Using fallback. Error: {e}"
                )
                formatted = f"{system_prompt}\n\n{p}" if system_prompt else p
                formatted_batch.append(formatted)
        
        if not formatted_batch:
            continue  # Skip batch if all prompts failed formatting
        
        # Tokenize formatted prompts
        inputs = tokenizer(
            formatted_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        ).to(device)
        
        hook.register_hooks(model)
        try:
            with torch.no_grad():
                model(**inputs)
        except Exception as e:
            logging.error(f"Model forward pass failed: {e}", exc_info=True)
            hook.remove_hooks()
            continue
        
        hook.remove_hooks()
        
        batch_cache = hook.get_cache()
        actual_batch_size = len(formatted_batch)
        
        for layer_idx in layers_to_cache:
            if (
                layer_idx in batch_cache
                and batch_cache[layer_idx].shape[0] == actual_batch_size
            ):
                all_activations[layer_idx].append(
                    batch_cache[layer_idx].detach().clone().float().cpu()
                )
            else:
                logging.warning(
                    f"Activation cache issue for layer {layer_idx}, batch {i}. "
                    f"Filling with zeros."
                )
                all_activations[layer_idx].append(
                    torch.zeros(
                        (actual_batch_size, d_model), dtype=torch.float32, device="cpu"
                    )
                )
    
    # Concatenate activations for each layer
    final_activations: Dict[int, Tensor] = {}
    
    for layer_idx in layers_to_cache:
        if all_activations[layer_idx]:
            try:
                final_activations[layer_idx] = torch.cat(
                    all_activations[layer_idx], dim=0
                )
            except RuntimeError as e:
                logging.error(
                    f"Error concatenating activations for layer {layer_idx}: {e}"
                )
                shapes = [t.shape for t in all_activations[layer_idx]]
                logging.error(f"Tensor shapes: {shapes}")
                
                # Try to find the most common shape and pad the rest
                if all(len(s) == 2 for s in shapes):
                    most_common_dim = max(set(s[1] for s in shapes), key=[s[1] for s in shapes].count)
                    padded_tensors = []
                    
                    for t in all_activations[layer_idx]:
                        if t.shape[1] != most_common_dim:
                            padding = torch.zeros(
                                (t.shape[0], most_common_dim - t.shape[1]), 
                                dtype=t.dtype, device=t.device
                            )
                            padded_t = torch.cat([t, padding], dim=1)
                            padded_tensors.append(padded_t)
                        else:
                            padded_tensors.append(t)
                    
                    try:
                        final_activations[layer_idx] = torch.cat(padded_tensors, dim=0)
                        logging.warning(
                            f"Successfully padded and concatenated tensors for layer {layer_idx}"
                        )
                    except RuntimeError:
                        logging.error(
                            f"Failed to concatenate even after padding for layer {layer_idx}"
                        )
                        # Fallback: create tensor with zeros
                        total_samples = sum(t.shape[0] for t in all_activations[layer_idx])
                        final_activations[layer_idx] = torch.zeros(
                            (total_samples, most_common_dim), dtype=torch.float32
                        )
    
    # Log the output shape for verification
    total_samples = len(prompts)
    for layer_idx, activations in final_activations.items():
        if activations.shape[0] != total_samples:
            logging.warning(
                f"Activation count mismatch for layer {layer_idx}. "
                f"Expected {total_samples}, got {activations.shape[0]}."
            )
    
    return final_activations


def prepare_controller_inputs(
    model: Any,
    tokenizer: Any,
    harmful_prompts: List[str],
    benign_prompts: Optional[List[str]] = None,
    controller_input_layers: List[int] = None,
    system_prompt: str = "",
    batch_size: int = 4,
    controller_input_pos: int = -1,
    device: Union[str, torch.device] = "cpu",
    llm_dtype: torch.dtype = torch.float32,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Prepare inputs for the controller by caching activations from the LLM.
    
    Args:
        model (Any): The LLM model
        tokenizer (Any): The tokenizer
        harmful_prompts (List[str]): List of harmful prompts
        benign_prompts (Optional[List[str]]): List of benign prompts, if any
        controller_input_layers (List[int]): Layer indices to use as controller input
        system_prompt (str): System prompt to prepend
        batch_size (int): Batch size for processing
        controller_input_pos (int): Position in sequence to take activations from
        device (Union[str, torch.device]): Device to run on
        llm_dtype (torch.dtype): Data type for the LLM
        
    Returns:
        Tuple[Tensor, Optional[Tensor]]: 
            - X_harmful: Tensor of concatenated activations for harmful prompts
            - X_benign: Tensor of concatenated activations for benign prompts, or None
    """
    # Cache activations for harmful prompts
    logging.info(f"Caching activations for {len(harmful_prompts)} harmful prompts...")
    
    harmful_activations = get_activation_cache(
        model=model,
        tokenizer=tokenizer,
        prompts=harmful_prompts,
        layers_to_cache=controller_input_layers,
        position_idx=controller_input_pos,
        batch_size=batch_size,
        system_prompt=system_prompt,
        device=device,
        dtype=llm_dtype,
    )
    
    # Validate harmful activations
    if harmful_activations and controller_input_layers:
        num_cached = harmful_activations[controller_input_layers[0]].shape[0]
        logging.info(f"Number of harmful activation vectors cached: {num_cached}")
        if num_cached == 0:
            raise ValueError("No harmful activations were cached. Cannot proceed with training.")
    else:
        raise ValueError("No harmful activations cached. Cannot proceed with training.")
    
    # Create controller inputs for harmful prompts
    harmful_input_list = [
        harmful_activations[l].float() for l in controller_input_layers
    ]
    X_harmful = torch.cat(harmful_input_list, dim=-1)
    
    # Cache activations for benign prompts if provided
    X_benign = None
    if benign_prompts and len(benign_prompts) > 0:
        logging.info(f"Caching activations for {len(benign_prompts)} benign prompts...")
        
        benign_activations = get_activation_cache(
            model=model,
            tokenizer=tokenizer,
            prompts=benign_prompts,
            layers_to_cache=controller_input_layers,
            position_idx=controller_input_pos,
            batch_size=batch_size,
            system_prompt=system_prompt,
            device=device,
            dtype=llm_dtype,
        )
        
        # Validate benign activations
        if benign_activations and controller_input_layers:
            num_benign_cached = benign_activations[controller_input_layers[0]].shape[0]
            logging.info(f"Number of benign activation vectors cached: {num_benign_cached}")
            if num_benign_cached > 0:
                # Create controller inputs for benign prompts
                benign_input_list = [
                    benign_activations[l].float() for l in controller_input_layers
                ]
                X_benign = torch.cat(benign_input_list, dim=-1)
            else:
                logging.warning("No benign activations were cached. Falling back to non-discriminative training.")
        else:
            logging.warning("Failed to cache benign activations. Falling back to non-discriminative training.")
    
    return X_harmful, X_benign
