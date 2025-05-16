"""
Activation caching utilities for extracting hidden states from LLM layers.

This module provides hooks and utilities to extract activations from LLM layers,
which are used for training the steering controller.
"""

import logging
from typing import Dict, List, Tuple, Optional, Union, Any

import torch
from torch import Tensor, nn


class ActivationCacheHook:
    """Hook for capturing activations from specific layers of an LLM model.
    
    This class implements hooks to extract and store activations from 
    specified layers during the forward pass of an LLM. It's primarily
    used during controller training to collect activations for training
    examples.
    
    Attributes:
        layers_to_cache (List[int]): List of layer indices to capture activations from.
        position_idx (int): Position in sequence to capture (-1 for last token).
        cache (Dict[int, Tensor]): Storage for captured activations.
        handles (List): List of hook handles for removal.
    """
    
    def __init__(self, layers_to_cache: List[int], position_idx: int = -1):
        """Initialize the activation cache hook.
        
        Args:
            layers_to_cache (List[int]): Indices of layers to extract activations from.
            position_idx (int): Position in sequence to extract (-1 for last token).
        """
        self.layers_to_cache = layers_to_cache
        self.position_idx = position_idx
        self.cache: Dict[int, Tensor] = {}
        self.handles = []

    def _get_layer_module(self, model: Any, layer_idx: int) -> nn.Module:
        """Get the module for a specific layer index.
        
        Args:
            model (Any): The LLM model (typically a HuggingFace model).
            layer_idx (int): Index of the layer to retrieve.
            
        Returns:
            nn.Module: The layer module.
            
        Raises:
            AttributeError: If the layer module list cannot be found.
            IndexError: If the layer index is out of range.
        """
        layer_module_list = None
        
        # Handle different model architectures
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            # For models with model.layers structure (like Llama)
            layer_module_list = model.model.layers
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            # For models with transformer.h structure (like GPT-2)
            layer_module_list = model.transformer.h
        elif hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
            # For encoder models like BERT
            layer_module_list = model.encoder.layer
        
        if layer_module_list is None:
            raise AttributeError(
                f"Cannot find layer module list in model {type(model)}."
            )
        
        if not (0 <= layer_idx < len(layer_module_list)):
            raise IndexError(f"Layer index {layer_idx} out of range.")
            
        return layer_module_list[layer_idx]

    def _hook_fn(self, layer_idx: int, module: nn.Module, input: Tuple[Tensor, ...], 
                output: Union[Tensor, Tuple[Tensor, ...]]) -> None:
        """Hook function called during model forward pass.
        
        Args:
            layer_idx (int): Index of the layer being processed.
            module (nn.Module): The layer module.
            input (Tuple[Tensor, ...]): Input tensors to the layer (not used).
            output (Union[Tensor, Tuple[Tensor, ...]]): Output from the layer.
        """
        try:
            # Extract hidden states (first element for tuple outputs)
            hidden_states = output[0] if isinstance(output, tuple) else output
            
            # Determine which position to extract
            batch_size, seq_len, hidden_dim = hidden_states.shape
            pos = self.position_idx if self.position_idx >= 0 else seq_len + self.position_idx
            
            if 0 <= pos < seq_len:
                # Extract activations for the specified position from all batch items
                activations = hidden_states[:, pos, :].detach().clone()
                self.cache[layer_idx] = activations
            else:
                logging.warning(
                    f"Position index {self.position_idx} (resolved to {pos}) "
                    f"out of range for sequence length {seq_len}. "
                    f"Falling back to last token."
                )
                self.cache[layer_idx] = hidden_states[:, -1, :].detach().clone()
        except Exception as e:
            logging.error(f"Error in activation hook for layer {layer_idx}: {e}")
            # Create empty tensor as fallback
            if hasattr(module, "config") and hasattr(module.config, "hidden_size"):
                d_model = module.config.hidden_size
            else:
                # Estimate from output shape if available
                d_model = hidden_states.shape[-1] if 'hidden_states' in locals() else 768
                
            self.cache[layer_idx] = torch.zeros(
                (1, d_model), 
                device=next(module.parameters()).device,
                dtype=next(module.parameters()).dtype
            )

    def register_hooks(self, model: Any) -> None:
        """Register hooks on the specified layers of the model.
        
        Args:
            model (Any): The LLM model to hook into.
        """
        self.remove_hooks()  # Remove any existing hooks
        self.cache = {}  # Clear cache
        
        for layer_idx in self.layers_to_cache:
            try:
                layer_module = self._get_layer_module(model, layer_idx)
                
                # Register forward hook
                handle = layer_module.register_forward_hook(
                    lambda mod, inp, out, idx=layer_idx: self._hook_fn(idx, mod, inp, out)
                )
                self.handles.append(handle)
            except Exception as e:
                logging.error(f"Failed to register hook for layer {layer_idx}: {e}")

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def get_cache(self) -> Dict[int, Tensor]:
        """Get the cached activations.
        
        Returns:
            Dict[int, Tensor]: Dictionary mapping layer indices to activation tensors.
        """
        return self.cache


def get_refusal_direction(
    model: Any,
    tokenizer: Any,
    refusal_tokens: List[str],
    answer_tokens: List[str],
    device: Optional[Union[str, torch.device]] = None,
) -> Optional[Tensor]:
    """Calculate the refusal direction in embedding space.
    
    Computes a vector in the model's embedding space that points from
    "helpful" (answer) tokens toward "refusal" tokens. This direction
    is used to guide activations during steering.
    
    Args:
        model (Any): The LLM model.
        tokenizer (Any): The tokenizer for the model.
        refusal_tokens (List[str]): List of tokens/words indicating refusal.
        answer_tokens (List[str]): List of tokens/words indicating helpfulness.
        device (Optional[Union[str, torch.device]]): Device to use, or None to use model's device.
            
    Returns:
        Optional[Tensor]: The computed refusal direction vector, or None if computation failed.
    """
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)
        
    try:
        # Get embedding layer
        if hasattr(model, "get_input_embeddings"):
            embedding_layer = model.get_input_embeddings()
        elif hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
            embedding_layer = model.transformer.wte
        elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            embedding_layer = model.model.embed_tokens
        else:
            logging.error("Could not find embedding layer in model")
            return None
        
        # Tokenize refusal and answer tokens
        refusal_token_ids = []
        for token in refusal_tokens:
            ids = tokenizer.encode(token, add_special_tokens=False)
            refusal_token_ids.extend(ids)
        
        answer_token_ids = []
        for token in answer_tokens:
            ids = tokenizer.encode(token, add_special_tokens=False)
            answer_token_ids.extend(ids)
        
        logging.info(f"Using {len(refusal_token_ids)} refusal token IDs and {len(answer_token_ids)} answer token IDs")
        
        if not refusal_token_ids or not answer_token_ids:
            logging.error("No valid token IDs found for refusal or answer tokens")
            return None
        
        # Convert to tensors
        refusal_ids_tensor = torch.tensor(refusal_token_ids, dtype=torch.long, device=device)
        answer_ids_tensor = torch.tensor(answer_token_ids, dtype=torch.long, device=device)
        
        # Get embeddings
        with torch.no_grad():
            refusal_embeddings = embedding_layer(refusal_ids_tensor)
            answer_embeddings = embedding_layer(answer_ids_tensor)
        
        # Calculate centroids
        refusal_centroid = torch.mean(refusal_embeddings, dim=0)
        answer_centroid = torch.mean(answer_embeddings, dim=0)
        
        # Calculate direction from answer to refusal
        direction = refusal_centroid - answer_centroid
        
        # Normalize
        direction_norm = torch.norm(direction)
        if direction_norm > 0:
            normalized_direction = direction / direction_norm
        else:
            logging.warning("Refusal direction norm is 0, using unnormalized direction")
            normalized_direction = direction
        
        logging.info(f"Calculated refusal direction with norm {direction_norm:.4f}")
        
        return normalized_direction
        
    except Exception as e:
        logging.error(f"Error calculating refusal direction: {e}")
        return None
