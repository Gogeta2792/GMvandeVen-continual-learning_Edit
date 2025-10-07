"""
Attention Router for Modular Continual Learning

This module implements a task-agnostic attention router that scores modules
in a ModularLayer without requiring task labels. The router uses input-driven
routing similar to LMC (local gating) but implemented as a single router.

The router supports both local and global routing modes, with regularization
for stable attention patterns to prevent abrupt routing changes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Literal
from dataclasses import dataclass


@dataclass
class RouterConfig:
    """Configuration for AttentionRouter.
    
    Attributes:
        num_modules: Number of modules to route between (M)
        input_dim: Input dimension for local routing
        global_dim: Global context dimension (0 if not using global routing)
        hidden_dim: Hidden dimension in the router MLP
        use_global: Whether to use global routing mode
        entropy_coef: Coefficient for entropy regularization (encourage sparsity)
        drift_coef: Coefficient for L2 drift regularization (encourage stability)
        ema_momentum: Momentum for EMA of past attention weights
        device: Device to place the router on
    """
    num_modules: int
    input_dim: int
    global_dim: int = 0
    hidden_dim: int = 128
    use_global: bool = False
    entropy_coef: float = 0.01
    drift_coef: float = 0.1
    ema_momentum: float = 0.99
    device: str = 'cpu'
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.num_modules <= 0:
            raise ValueError("num_modules must be a positive integer")
        if self.input_dim <= 0:
            raise ValueError("input_dim must be a positive integer")
        if self.use_global and self.global_dim <= 0:
            raise ValueError("global_dim must be positive when use_global=True")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer")
        if not (0.0 <= self.entropy_coef <= 1.0):
            raise ValueError("entropy_coef must be in range [0.0, 1.0]")
        if not (0.0 <= self.drift_coef <= 1.0):
            raise ValueError("drift_coef must be in range [0.0, 1.0]")
        if not (0.0 < self.ema_momentum < 1.0):
            raise ValueError("ema_momentum must be in range (0.0, 1.0)")


class AttentionRouter(nn.Module):
    """Task-agnostic attention router for modular layers.
    
    This router computes attention weights over M modules based on input
    characteristics, without requiring task labels. It supports both local
    and global routing modes with regularization for stable attention.
    
    Architecture: LayerNorm → Linear → GELU → Linear → logits
    """
    
    def __init__(self, config: RouterConfig):
        super().__init__()
        
        self.config = config
        self.num_modules = config.num_modules
        self.use_global = config.use_global
        
        # Determine input dimension
        if self.use_global:
            self.input_dim = config.input_dim + config.global_dim
        else:
            self.input_dim = config.input_dim
        
        # Router MLP: LayerNorm → Linear → GELU → Linear → logits
        self.layer_norm = nn.LayerNorm(self.input_dim)
        self.linear1 = nn.Linear(self.input_dim, config.hidden_dim)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(config.hidden_dim, self.num_modules)
        
        # Initialize weights
        self._init_weights()
        
        # EMA for drift regularization
        self.register_buffer('ema_attention', torch.zeros(1, self.num_modules))
        self.ema_momentum = config.ema_momentum
        
        # Move to specified device
        self.to(config.device)
    
    def _init_weights(self):
        """Initialize router weights for stable training."""
        # Initialize linear layers with small random weights
        nn.init.xavier_uniform_(self.linear1.weight, gain=0.1)
        nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_uniform_(self.linear2.weight, gain=0.1)
        nn.init.zeros_(self.linear2.bias)
    
    def forward(self, x: torch.Tensor, global_context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass through the attention router.
        
        Args:
            x: Input tensor for local routing, shape [B, input_dim]
            global_context: Optional global context tensor, shape [B, global_dim]
            
        Returns:
            Tuple of:
            - logits: Raw attention logits, shape [B, num_modules]
            - attention: Softmax attention weights, shape [B, num_modules]
            - regularizers: Dictionary of regularization losses
        """
        batch_size = x.size(0)
        
        # Prepare input
        if self.use_global:
            if global_context is None:
                raise ValueError("global_context required when use_global=True")
            if global_context.size(0) != batch_size:
                raise ValueError("Batch size mismatch between x and global_context")
            router_input = torch.cat([x, global_context], dim=-1)
        else:
            router_input = x
        
        # Router forward pass: LayerNorm → Linear → GELU → Linear
        x_norm = self.layer_norm(router_input)
        x_hidden = self.gelu(self.linear1(x_norm))
        logits = self.linear2(x_hidden)
        
        # Convert to attention weights
        attention = F.softmax(logits, dim=-1)
        
        # Update EMA for drift regularization
        with torch.no_grad():
            if self.training:
                # Update EMA with current batch mean attention
                batch_mean_attention = attention.mean(dim=0, keepdim=True)
                self.ema_attention = (self.ema_momentum * self.ema_attention + 
                                    (1 - self.ema_momentum) * batch_mean_attention)
        
        # Compute regularizers
        regularizers = self._compute_regularizers(attention)
        
        return logits, attention, regularizers
    
    def _compute_regularizers(self, attention: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute regularization losses for stable attention.
        
        Args:
            attention: Attention weights, shape [B, num_modules]
            
        Returns:
            Dictionary containing regularization losses
        """
        regularizers = {}
        
        # Entropy penalty: encourage confident, sparse selections
        if self.config.entropy_coef > 0:
            # Compute entropy: -sum(p * log(p))
            entropy = -torch.sum(attention * torch.log(attention + 1e-8), dim=-1)
            # Penalty: higher entropy = more uniform = less confident
            entropy_penalty = self.config.entropy_coef * entropy.mean()
            regularizers['entropy_penalty'] = entropy_penalty
        
        # L2 drift penalty: discourage abrupt routing changes
        if self.config.drift_coef > 0 and self.ema_attention.numel() > 0:
            # Compare current batch mean with EMA
            current_mean = attention.mean(dim=0, keepdim=True)
            drift = F.mse_loss(current_mean, self.ema_attention)
            drift_penalty = self.config.drift_coef * drift
            regularizers['drift_penalty'] = drift_penalty
        
        return regularizers
    
    def reset_ema(self):
        """Reset the EMA attention buffer."""
        self.ema_attention.zero_()
    
    def get_attention_stats(self) -> Dict[str, float]:
        """Get statistics about current attention patterns.
        
        Returns:
            Dictionary with attention statistics
        """
        if self.ema_attention.numel() == 0:
            return {'ema_attention': None}
        
        ema_attn = self.ema_attention.squeeze()
        return {
            'ema_attention': ema_attn.cpu().numpy().tolist(),
            'ema_entropy': -torch.sum(ema_attn * torch.log(ema_attn + 1e-8)).item(),
            'ema_max_weight': ema_attn.max().item(),
            'ema_min_weight': ema_attn.min().item()
        }


class LocalAttentionRouter(AttentionRouter):
    """Convenience class for local-only attention routing."""
    
    def __init__(self, num_modules: int, input_dim: int, hidden_dim: int = 128,
                 entropy_coef: float = 0.01, drift_coef: float = 0.1, device: str = 'cpu'):
        config = RouterConfig(
            num_modules=num_modules,
            input_dim=input_dim,
            global_dim=0,
            hidden_dim=hidden_dim,
            use_global=False,
            entropy_coef=entropy_coef,
            drift_coef=drift_coef,
            device=device
        )
        super().__init__(config)


class GlobalAttentionRouter(AttentionRouter):
    """Convenience class for global attention routing."""
    
    def __init__(self, num_modules: int, input_dim: int, global_dim: int,
                 hidden_dim: int = 128, entropy_coef: float = 0.01, 
                 drift_coef: float = 0.1, device: str = 'cpu'):
        config = RouterConfig(
            num_modules=num_modules,
            input_dim=input_dim,
            global_dim=global_dim,
            hidden_dim=hidden_dim,
            use_global=True,
            entropy_coef=entropy_coef,
            drift_coef=drift_coef,
            device=device
        )
        super().__init__(config)


# Example usage functions
def create_local_router(num_modules: int, input_dim: int, **kwargs) -> LocalAttentionRouter:
    """Create a local attention router."""
    return LocalAttentionRouter(num_modules, input_dim, **kwargs)


def create_global_router(num_modules: int, input_dim: int, global_dim: int, **kwargs) -> GlobalAttentionRouter:
    """Create a global attention router."""
    return GlobalAttentionRouter(num_modules, input_dim, global_dim, **kwargs)
