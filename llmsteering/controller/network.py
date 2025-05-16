"""
Controller network implementation for LLM Safety Steering.

This module defines the controller network architecture used to steer
LLM activations toward refusal responses for harmful prompts.
"""

import logging
from typing import Dict, List, Tuple, Optional, Union, Any

import torch
from torch import Tensor, nn
import functools


class SteeringController(nn.Module):
    """Controller network that outputs both a scalar value and layer weights.
    
    The controller network is the core component of the LLM Safety Steering system.
    It takes activations from select layers of the LLM and outputs:
    1. A scalar value indicating whether a steering intervention is needed
    2. Per-layer weights determining how much each layer should be modified
    
    Attributes:
        num_layers (int): The number of layers in the target LLM.
        output_dim (int): The dimension of the output (1 + num_layers).
        fc1 (nn.Linear): First fully connected layer.
        relu (nn.ReLU): ReLU activation function.
        fc2 (nn.Linear): Second fully connected layer.
    """
    
    def __init__(self, input_dim: int, num_layers: int, hidden_dim: int = 1024):
        """Initialize the controller network.
        
        Args:
            input_dim (int): Dimension of the input, typically combined activations
                from selected layers of the LLM.
            num_layers (int): Number of layers in the target LLM.
            hidden_dim (int): Hidden dimension of the controller network.
        """
        super().__init__()
        self.num_layers = num_layers
        self.output_dim = 1 + num_layers  # 1 for scalar, num_layers for weights
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, self.output_dim)
        logging.info(
            f"Initialized SteeringController: InputDim={input_dim}, "
            f"HiddenDim={hidden_dim}, OutputDim={self.output_dim} "
            f"(1 scalar + {num_layers} weights)"
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass through the controller network.
        
        Args:
            x (Tensor): Input tensor of shape [batch_size, input_dim].
                Typically contains concatenated activations from select LLM layers.
                
        Returns:
            Tuple[Tensor, Tensor]: 
                - scalar_out: Scalar steering intensity, shape [batch_size, 1]
                - weight_logits: Layer weight logits, shape [batch_size, num_layers]
        """
        # Ensure input is on the same device as weights
        device = self.fc1.weight.device
        x = x.to(device)
        x = self.fc1(x.to(torch.float32))
        x = self.relu(x)
        raw_output = self.fc2(x)
        scalar_out = raw_output[:, 0:1]
        weight_logits = raw_output[:, 1:]
        return scalar_out, weight_logits
    
    @classmethod
    def from_pretrained(cls, path: str, num_layers: Optional[int] = None, 
                      input_dim: Optional[int] = None, hidden_dim: Optional[int] = None) -> "SteeringController":
        """Load a pretrained controller from a checkpoint file.
        
        Args:
            path (str): Path to the saved controller checkpoint.
            num_layers (int, optional): Number of layers in the target LLM. If None,
                will be inferred from the checkpoint.
            input_dim (int, optional): Dimension of the input. If None, will be
                inferred from the checkpoint.
            hidden_dim (int, optional): Hidden dimension. If None, will be inferred
                from the checkpoint.
                
        Returns:
            SteeringController: Loaded controller instance.
            
        Raises:
            ValueError: If required dimensions cannot be inferred from the checkpoint.
        """
        checkpoint = torch.load(path, map_location='cpu')
        
        # If this is a state dict only
        if isinstance(checkpoint, dict) and 'fc1.weight' in checkpoint:
            # Infer dimensions from the state dict
            if input_dim is None:
                input_dim = checkpoint['fc1.weight'].shape[1]
            if num_layers is None:
                num_layers = checkpoint['fc2.weight'].shape[0] - 1  # -1 for scalar output
            if hidden_dim is None:
                hidden_dim = checkpoint['fc1.weight'].shape[0]
                
            controller = cls(input_dim=input_dim, num_layers=num_layers, hidden_dim=hidden_dim)
            controller.load_state_dict(checkpoint)
        else:
            # Support for older format or custom saving approach
            raise ValueError("Unsupported checkpoint format. Expected a state dict.")
            
        return controller
    
    def save(self, path: str) -> None:
        """Save the controller to a file.
        
        Args:
            path (str): Path where to save the controller.
        """
        torch.save(self.state_dict(), path)
        logging.info(f"Saved controller to {path}")


class SteeringModifier:
    """Manages the modification of LLM activations using a SteeringController.
    
    This class implements the hooks needed to:
    1. Capture activations from selected layers of the LLM
    2. Run the controller to determine modification parameters
    3. Apply the modifications to the appropriate LLM layers
    
    Attributes:
        controller (SteeringController): The controller network
        refusal_dir (Tensor): The refusal direction in the embedding space
        input_layers (List[int]): Indices of layers to use as input to the controller
        num_llm_layers (int): Total number of layers in the LLM
        scale_factor (float): Scaling factor for the modifications
        input_pos (int): Position in the sequence to use for controller input
        apply_pos (int): Position in the sequence to apply modifications
        device (torch.device): Device to run the controller on
        llm_dtype (torch.dtype): Data type of the LLM
        controller_dtype (torch.dtype): Data type of the controller
        handles (List): Hook handles for easy removal
        captured_inputs (Dict): Storage for captured activations
        active_batch_size (int): Batch size for the current generation
        step_scalar (Dict): Scalar values for the current generation step
        step_weights (Dict): Weight values for the current generation step
        controller_run_for_step (Dict): Tracking if controller has run for the step
    """
    
    def __init__(
        self,
        controller: SteeringController,
        refusal_direction: Tensor,
        controller_input_layers: List[int],
        num_llm_layers: int,
        patch_scale_factor: float = 1.0,
        input_pos: int = -1,
        apply_pos: int = -1,
        device: Union[str, torch.device] = "cpu",
        llm_dtype: torch.dtype = torch.float32,
        controller_dtype: torch.dtype = torch.float32,
    ):
        """Initialize the steering modifier.
        
        Args:
            controller (SteeringController): The controller network
            refusal_direction (Tensor): The refusal direction in embedding space
            controller_input_layers (List[int]): Indices of layers to use as input
            num_llm_layers (int): Total number of layers in the LLM
            patch_scale_factor (float): Scaling factor for modifications
            input_pos (int): Position in sequence to use for controller input (-1 for last)
            apply_pos (int): Position in sequence to apply modifications (-1 for last)
            device (Union[str, torch.device]): Device to run on
            llm_dtype (torch.dtype): Data type of the LLM
            controller_dtype (torch.dtype): Data type of the controller
        """
        # Convert string device to torch.device
        device = torch.device(device)
        
        # Move controller to the specified device and dtype
        self.controller = controller.to(device=device, dtype=controller_dtype)
        self.controller.eval()
        
        # Move refusal direction to the same device (but keep original dtype)
        self.refusal_dir = refusal_direction.to(device=device)
        
        self.input_layers = sorted(controller_input_layers)
        self.num_llm_layers = num_llm_layers
        self.scale_factor = patch_scale_factor
        self.input_pos = input_pos
        self.apply_pos = apply_pos
        self.device = device
        self.llm_dtype = llm_dtype
        self.controller_dtype = controller_dtype
        self.handles = []
        self.captured_inputs: Dict[Tuple[int, int], Tensor] = {}
        self.active_batch_size = 0
        self.step_scalar: Dict[int, Tensor] = {}
        self.step_weights: Dict[int, Tensor] = {}
        self.controller_run_for_step: Dict[int, bool] = {}

    def _get_layer_module(self, model: Any, layer_idx: int) -> nn.Module:
        """Get the layer module from the model at the specified index.
        
        Args:
            model (Any): The LLM model (typically a transformers model)
            layer_idx (int): Index of the layer to retrieve
            
        Returns:
            nn.Module: The layer module
            
        Raises:
            AttributeError: If the layer module list cannot be found
            IndexError: If the layer index is out of range
        """
        layer_module_list = None
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layer_module_list = model.model.layers
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            layer_module_list = model.transformer.h
        if layer_module_list is None:
            raise AttributeError(
                f"Cannot find layer module list in model {type(model)}."
            )
        if not (0 <= layer_idx < len(layer_module_list)):
            raise IndexError(f"Layer index {layer_idx} out of range.")
        return layer_module_list[layer_idx]

    def capture_hook_fn(
        self,
        layer_idx: int,
        module: nn.Module,
        input: Tuple[Tensor],
        output: Union[Tuple[Tensor], Tensor],
    ):
        """Hook to capture activations from a layer.
        
        Args:
            layer_idx (int): Index of the layer being processed
            module (nn.Module): The layer module
            input (Tuple[Tensor]): Input to the layer (not used)
            output (Union[Tuple[Tensor], Tensor]): Output from the layer
        """
        hidden_state = output[0] if isinstance(output, tuple) else output
        current_batch_size = hidden_state.shape[0]
        if current_batch_size != self.active_batch_size:
            self.captured_inputs = {}
            self.active_batch_size = current_batch_size
        try:
            for b_idx in range(self.active_batch_size):
                seq_len = hidden_state.shape[1]
                pos_idx = (
                    self.input_pos if self.input_pos >= 0 else seq_len + self.input_pos
                )
                if 0 <= pos_idx < seq_len:
                    self.captured_inputs[(b_idx, layer_idx)] = (
                        hidden_state[b_idx, pos_idx, :].detach().clone()
                    )
                else:
                    if (b_idx, layer_idx) in self.captured_inputs:
                        del self.captured_inputs[(b_idx, layer_idx)]
        except Exception as e:
            logging.debug(
                f"Capture hook error L{layer_idx}: {e}", exc_info=False
            )  # Less verbose debug

    def apply_hook_fn(
        self,
        current_layer_idx: int,
        module: nn.Module,
        input: Tuple[Tensor],
        output: Union[Tuple[Tensor], Tensor],
    ):
        """Hook to apply modifications to activations based on controller output.
        
        Args:
            current_layer_idx (int): Index of the layer being processed
            module (nn.Module): The layer module
            input (Tuple[Tensor]): Input to the layer (not used)
            output (Union[Tuple[Tensor], Tensor]): Output from the layer
            
        Returns:
            Union[Tuple[Tensor], Tensor]: Modified output
        """
        hidden_states_orig = output[0] if isinstance(output, tuple) else output
        current_batch_size = hidden_states_orig.shape[0]
        if current_batch_size != self.active_batch_size:
            return output

        modified_hidden_states_list = []
        apply_occurred_batch = False

        for b_idx in range(current_batch_size):
            if not self.controller_run_for_step.get(b_idx, False):
                inputs_ready = all(
                    (b_idx, l) in self.captured_inputs for l in self.input_layers
                )
                if inputs_ready:
                    try:
                        controller_input_list = [
                            self.captured_inputs[(b_idx, l)] for l in self.input_layers
                        ]
                        controller_input_llm_dtype = torch.cat(
                            controller_input_list, dim=-1
                        ).unsqueeze(0)
                        
                        # The controller.forward() handles device/dtype conversion
                        controller_input = controller_input_llm_dtype
                        
                        with torch.no_grad():
                            scalar_out_raw, weight_logits = self.controller(
                                controller_input
                            )
                        self.step_scalar[b_idx] = scalar_out_raw.squeeze()
                        self.step_weights[b_idx] = torch.sigmoid(
                            weight_logits
                        ).squeeze()
                        self.controller_run_for_step[b_idx] = True
                        for l in self.input_layers:
                            if (b_idx, l) in self.captured_inputs:
                                del self.captured_inputs[(b_idx, l)]
                    except Exception as e:
                        logging.error(
                            f"Controller exec failed b{b_idx}: {e}", exc_info=True
                        )
                        self.controller_run_for_step[b_idx] = False
                else:
                    self.controller_run_for_step[b_idx] = False

            current_hidden_state_item = hidden_states_orig[b_idx].clone()
            if self.controller_run_for_step.get(b_idx, False):
                try:
                    layer_weight = self.step_weights[b_idx][current_layer_idx].to(
                        self.llm_dtype
                    )
                    scalar_magnitude = self.step_scalar[b_idx].to(self.llm_dtype)
                    
                    # Get the device from the current hidden state
                    current_device = current_hidden_state_item.device
                    
                    # Ensure refusal direction is on the same device and dtype
                    refusal_dir_typed = self.refusal_dir.to(dtype=self.llm_dtype, device=current_device)
                    
                    # Ensure scalar and layer weight are also on the same device
                    layer_weight = layer_weight.to(device=current_device)
                    scalar_magnitude = scalar_magnitude.to(device=current_device)
                    
                    base_patch = (
                        scalar_magnitude * refusal_dir_typed * self.scale_factor
                    )
                    weighted_patch = base_patch * layer_weight
                    seq_len = current_hidden_state_item.shape[0]
                    apply_pos_idx = (
                        self.apply_pos
                        if self.apply_pos >= 0
                        else seq_len + self.apply_pos
                    )
                    if 0 <= apply_pos_idx < seq_len:
                        current_activation = current_hidden_state_item[apply_pos_idx, :]
                        modified_activation = current_activation + weighted_patch
                        current_hidden_state_item[apply_pos_idx, :] = (
                            modified_activation
                        )
                        apply_occurred_batch = True
                except Exception as e:
                    logging.error(
                        f"Patch apply failed L{current_layer_idx} b{b_idx}: {e}",
                        exc_info=True,
                    )

            modified_hidden_states_list.append(current_hidden_state_item)

            if current_layer_idx == self.num_llm_layers - 1:
                self.controller_run_for_step[b_idx] = False
                if b_idx in self.step_scalar:
                    del self.step_scalar[b_idx]
                if b_idx in self.step_weights:
                    del self.step_weights[b_idx]
                for l in self.input_layers:
                    if (b_idx, l) in self.captured_inputs:
                        del self.captured_inputs[(b_idx, l)]

        if len(modified_hidden_states_list) != current_batch_size:
            modified_hidden_states = hidden_states_orig
        elif apply_occurred_batch:
            modified_hidden_states = torch.stack(modified_hidden_states_list, dim=0)
        else:
            modified_hidden_states = hidden_states_orig  # Avoid unnecessary stack if no modification happened

        if isinstance(output, tuple):
            return (modified_hidden_states,) + output[1:]
        else:
            return modified_hidden_states

    def register_hooks(self, model: Any):
        """Register hooks on the model to capture and modify activations.
        
        Args:
            model (Any): The LLM model
        """
        self.remove_hooks()
        self.captured_inputs = {}
        self.active_batch_size = 0
        self.step_scalar = {}
        self.step_weights = {}
        self.controller_run_for_step = {}
        for layer_idx in self.input_layers:
            try:
                layer_module = self._get_layer_module(model, layer_idx)
                handle = layer_module.register_forward_hook(
                    functools.partial(self.capture_hook_fn, layer_idx)
                )
                self.handles.append(handle)
            except Exception as e:
                logging.error(
                    f"Failed capture hook layer {layer_idx}: {e}", exc_info=True
                )
                raise
        for layer_idx in range(self.num_llm_layers):
            try:
                layer_module = self._get_layer_module(model, layer_idx)
                handle = layer_module.register_forward_hook(
                    functools.partial(self.apply_hook_fn, layer_idx)
                )
                self.handles.append(handle)
            except Exception as e:
                logging.error(
                    f"Failed apply hook layer {layer_idx}: {e}", exc_info=True
                )
                raise
        logging.debug(
            f"Registered {len(self.handles)} hooks ({len(self.input_layers)} capture, {self.num_llm_layers} apply)"
        )

    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles = []
