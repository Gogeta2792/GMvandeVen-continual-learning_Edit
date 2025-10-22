"""
Module expansion logic for dynamic network growth.
"""
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Dict, Any
import math


@dataclass
class ExpansionEvent:
    """Record of a module expansion event."""
    task_id: int
    layer_name: str
    new_module_id: int
    params_added: int
    confidence_score: float


class ExpansionTracker:
    """
    Tracks confidence scores and triggers module expansion.
    
    This class maintains moving averages of routing confidence scores
    and determines when to trigger expansion based on thresholds.
    """
    
    def __init__(
        self,
        threshold: float = 0.2,
        cooldown_tasks: int = 1,
        max_modules: int = 1,
        initial_modules: int = 1
    ):
        """
        Args:
            threshold: Confidence threshold below which expansion is triggered
            cooldown_tasks: Number of tasks to wait after expansion
            max_modules: Maximum modules allowed per layer
            initial_modules: Initial number of modules
        """
        self.threshold = threshold
        self.cooldown_tasks = cooldown_tasks
        self.max_modules = max_modules
        
        self.current_modules = initial_modules
        self.tasks_since_expansion = cooldown_tasks  # Allow immediate first expansion
        self.expansion_history = []
        
        # Track confidence scores
        self.confidence_buffer = []
        self.buffer_size = 100  # Keep last 100 confidence values
    
    def update_confidence(self, confidence: float):
        """
        Update the confidence score buffer.
        
        Args:
            confidence: New confidence score (0-1)
        """
        self.confidence_buffer.append(confidence)
        if len(self.confidence_buffer) > self.buffer_size:
            self.confidence_buffer.pop(0)
    
    def get_average_confidence(self) -> float:
        """Get the moving average confidence."""
        if len(self.confidence_buffer) == 0:
            return 1.0
        return sum(self.confidence_buffer) / len(self.confidence_buffer)
    
    def check_trigger(self, layer_stats: Optional[Dict[str, float]] = None) -> bool:
        """
        Check if expansion should be triggered.
        
        Args:
            layer_stats: Optional dictionary with layer-specific stats
        
        Returns:
            True if expansion should occur
        """
        # Check if we have room for more modules
        if self.current_modules >= self.max_modules:
            return False
        
        # Check cooldown period
        if self.tasks_since_expansion < self.cooldown_tasks:
            return False
        
        # Check confidence threshold
        avg_confidence = self.get_average_confidence()
        if avg_confidence >= self.threshold:
            return False
        
        return True
    
    def reset_after_task(self):
        """Reset confidence buffer after each task."""
        self.confidence_buffer = []
        self.tasks_since_expansion += 1


def apply_expansion(
    model: nn.Module,
    layer_name: str,
    task_id: int,
    confidence: float
) -> ExpansionEvent:
    """
    Apply module expansion to a specific layer.
    
    Args:
        model: The model to expand
        layer_name: Name of the layer to expand
        task_id: Current task ID
        confidence: Confidence score that triggered expansion
    
    Returns:
        ExpansionEvent documenting the expansion
    """
    # This is a simplified expansion - actual implementation depends on model architecture
    # For modular layers with built-in expansion support:
    
    layer = getattr(model, layer_name, None)
    if layer is None:
        raise ValueError(f"Layer {layer_name} not found in model")
    
    # Count parameters before expansion
    params_before = sum(p.numel() for p in layer.parameters())
    
    # If the layer has a maybe_expand method (as in ModularLayer)
    if hasattr(layer, 'maybe_expand'):
        expansion_result = layer.maybe_expand(
            attention=torch.ones(1, layer.num_modules) * confidence,
            force=True,
            verbose=False
        )
        new_module_id = layer.num_modules - 1
    else:
        # Generic expansion not supported
        raise NotImplementedError(f"Layer {layer_name} does not support expansion")
    
    # Count parameters after expansion
    params_after = sum(p.numel() for p in layer.parameters())
    params_added = params_after - params_before
    
    return ExpansionEvent(
        task_id=task_id,
        layer_name=layer_name,
        new_module_id=new_module_id,
        params_added=params_added,
        confidence_score=confidence
    )


def compute_confidence_from_attention(
    attention: torch.Tensor,
    method: str = 'max_weight'
) -> float:
    """
    Compute confidence score from attention weights.
    
    Args:
        attention: Attention weights, shape [batch, num_modules]
        method: Method to compute confidence ('max_weight' or 'entropy')
    
    Returns:
        Confidence score (0-1)
    """
    if method == 'max_weight':
        # Max attention weight across batch
        confidence = attention.max(dim=-1)[0].mean().item()
    
    elif method == 'entropy':
        # Inverse normalized entropy
        num_modules = attention.size(-1)
        entropy = -torch.sum(attention * torch.log(attention + 1e-8), dim=-1)
        max_entropy = math.log(num_modules)
        normalized_entropy = entropy / (max_entropy + 1e-8)
        confidence = (1.0 - normalized_entropy).mean().item()
    
    else:
        raise ValueError(f"Unknown confidence method: {method}")
    
    return confidence

