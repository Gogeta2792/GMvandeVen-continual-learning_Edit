"""
Modular Continual Learning Layer Implementation

This module implements a modular layer system for continual learning where each layer
contains multiple parallel sub-modules that can be selectively activated via attention
mechanisms. This enables task-agnostic subnetwork routing and skill reuse across tasks.

The ModularLayer class is designed to work with the existing continual learning framework
and will later be integrated with attention-based routing mechanisms.
"""

import torch
import torch.nn as nn
from torch import Tensor
from dataclasses import dataclass
from typing import Literal, Optional, Tuple, List, Dict
import math


@dataclass
class ModularLayerConfig:
    """Configuration for ModularLayer.
    
    This dataclass defines the parameters for creating a modular layer with
    multiple parallel sub-modules. The configuration supports both MLP and
    convolutional block types for different network architectures.
    
    Attributes:
        in_dim: Input dimension for the layer
        out_dim: Output dimension for the layer  
        num_modules: Number of parallel sub-modules (M)
        block_type: Type of sub-module ('mlp' or 'conv')
        hidden_dim: Hidden dimension for MLP blocks (defaults to out_dim)
        bias: Whether to use bias in linear/conv layers
        dropout: Dropout probability (0.0 = no dropout)
        batch_norm: Whether to use batch normalization
        use_router: Whether to use attention routing (if False, returns all outputs)
        router_hidden_dim: Hidden dimension for the attention router
        router_entropy_coef: Entropy regularization coefficient for router
        router_drift_coef: Drift regularization coefficient for router
        device: Device to place the layer on
    """
    in_dim: int
    out_dim: int
    num_modules: int = 4
    block_type: Literal['mlp', 'conv'] = 'mlp'
    hidden_dim: Optional[int] = None
    bias: bool = True
    dropout: float = 0.0
    batch_norm: bool = True
    use_router: bool = False
    router_hidden_dim: int = 128
    router_entropy_coef: float = 0.01
    router_drift_coef: float = 0.1
    device: str = 'cpu'
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.in_dim <= 0 or self.out_dim <= 0:
            raise ValueError("in_dim and out_dim must be positive integers")
        if self.num_modules <= 0:
            raise ValueError("num_modules must be a positive integer")
        if self.block_type not in ['mlp', 'conv']:
            raise ValueError("block_type must be 'mlp' or 'conv'")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("dropout must be in range [0.0, 1.0)")
        if self.hidden_dim is None:
            self.hidden_dim = self.out_dim
        if self.router_hidden_dim <= 0:
            raise ValueError("router_hidden_dim must be a positive integer")
        if not (0.0 <= self.router_entropy_coef <= 1.0):
            raise ValueError("router_entropy_coef must be in range [0.0, 1.0]")
        if not (0.0 <= self.router_drift_coef <= 1.0):
            raise ValueError("router_drift_coef must be in range [0.0, 1.0]")


class MLPBlock(nn.Module):
    """MLP sub-module: Linear → ReLU → Linear.
    
    This block implements a two-layer MLP with ReLU activation between layers.
    The block is designed to be used as a sub-module within the ModularLayer.
    
    Args:
        in_dim: Input dimension
        hidden_dim: Hidden layer dimension
        out_dim: Output dimension
        bias: Whether to use bias terms
        dropout: Dropout probability
    """
    
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, 
                 bias: bool = True, dropout: float = 0.0):
        super().__init__()
        
        self.linear1 = nn.Linear(in_dim, hidden_dim, bias=bias)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, out_dim, bias=bias)
        
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
            
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the MLP block.
        
        Args:
            x: Input tensor of shape [batch_size, in_dim]
            
        Returns:
            Output tensor of shape [batch_size, out_dim]
        """
        x = self.linear1(x)
        x = self.relu(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.linear2(x)
        return x


class ConvBlock(nn.Module):
    """Convolutional sub-module: Conv3×3 → BN → ReLU.
    
    This block implements a convolutional layer with batch normalization
    and ReLU activation. Designed for image processing tasks like CIFAR.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        bias: Whether to use bias in convolution
        dropout: Dropout probability
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 bias: bool = True, dropout: float = 0.0):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                             padding=1, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
        if dropout > 0.0:
            self.dropout = nn.Dropout2d(dropout)
        else:
            self.dropout = None
            
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the convolutional block.
        
        Args:
            x: Input tensor of shape [batch_size, in_channels, H, W]
            
        Returns:
            Output tensor of shape [batch_size, out_channels, H, W]
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class ModularLayer(nn.Module):
    """Modular layer with multiple parallel sub-modules and optional attention routing.
    
    This layer contains M parallel sub-modules that can be selectively activated
    via attention mechanisms. When use_router=True, it uses an attention router
    to compute weighted combinations of module outputs. When use_router=False,
    it returns all module outputs stacked along the module dimension.
    
    The layer is designed for continual learning scenarios where different modules
    can specialize for different tasks while sharing common representations.
    
    Args:
        config: Configuration object defining layer parameters
    """
    
    def __init__(self, config: ModularLayerConfig):
        super().__init__()
        
        self.config = config
        self.num_modules = config.num_modules
        self.block_type = config.block_type
        self.use_router = config.use_router
        
        # Create the parallel sub-modules
        self.modules_list = nn.ModuleList()
        
        if config.block_type == 'mlp':
            for i in range(config.num_modules):
                block = MLPBlock(
                    in_dim=config.in_dim,
                    hidden_dim=config.hidden_dim,
                    out_dim=config.out_dim,
                    bias=config.bias,
                    dropout=config.dropout
                )
                self.modules_list.append(block)
                
        elif config.block_type == 'conv':
            for i in range(config.num_modules):
                block = ConvBlock(
                    in_channels=config.in_dim,  # in_dim represents channels for conv
                    out_channels=config.out_dim,  # out_dim represents channels for conv
                    bias=config.bias,
                    dropout=config.dropout
                )
                self.modules_list.append(block)
        
        # Create attention router if requested
        if self.use_router:
            from models.attention_router import LocalAttentionRouter
            # For conv layers, we need to handle the input dimension differently
            # We'll create the router with the appropriate input dimension
            if config.block_type == 'conv':
                # For conv layers, we'll use a reasonable default input dimension
                # The actual input will be flattened in the forward pass
                router_input_dim = config.in_dim * 32 * 32  # Assume 32x32 images by default
            else:
                router_input_dim = config.in_dim
            
            self.router = LocalAttentionRouter(
                num_modules=config.num_modules,
                input_dim=router_input_dim,
                hidden_dim=config.router_hidden_dim,
                entropy_coef=config.router_entropy_coef,
                drift_coef=config.router_drift_coef,
                device=config.device
            )
        else:
            self.router = None
        
        # Move to specified device
        self.to(config.device)
        
    def forward(self, x: Tensor, global_ctx: Optional[Tensor] = None, 
                return_attn: bool = False, prune_threshold: float = 0.0) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        """Forward pass through sub-modules with attention-weighted routing.
        
        This is the core forward rule: compute all module outputs, get per-sample
        weights from the router, then take a weighted sum along the module axis
        (like a soft Mixture-of-Experts).
        
        Args:
            x: Input tensor
                - For MLP: [batch_size, in_dim]
                - For Conv: [batch_size, in_channels, H, W]
            global_ctx: Optional global context tensor for global routing
            return_attn: Whether to return attention weights and regularizers
            prune_threshold: Threshold for pruning small attention weights (0.0 = no pruning)
                
        Returns:
            If use_router=True:
                - output: Attention-weighted combination, shape [batch_size, out_dim] or [batch_size, out_channels, H, W]
                - attention_info: Dict with 'attention', 'logits', 'regularizers' (if return_attn=True)
            If use_router=False:
                - output: All module outputs stacked, shape [batch_size, num_modules, out_dim] or [batch_size, num_modules, out_channels, H, W]
                - attention_info: None
        """
        batch_size = x.size(0)
        
        # Compute outputs from all modules
        module_outputs = []
        for module in self.modules_list:
            output = module(x)
            module_outputs.append(output)
        
        if self.use_router and self.router is not None:
            # Use attention routing
            if self.block_type == 'conv':
                # For conv layers, we need to flatten the input for the router
                batch_size, channels, height, width = x.shape
                x_flat = x.view(batch_size, -1)  # [batch_size, channels*height*width]
                logits, attention, regularizers = self.router(x_flat, global_ctx)
            else:  # mlp
                logits, attention, regularizers = self.router(x, global_ctx)
            
            # Ensure numerical stability
            attention = torch.clamp(attention, min=1e-8, max=1.0)
            attention = attention.float()  # Ensure float32
            
            # Apply pruning if requested (for speed optimization)
            if prune_threshold > 0.0:
                attention = self._prune_attention(attention, prune_threshold)
            
            # Stack module outputs: mods = stack([mod_i(x)])
            if self.block_type == 'mlp':
                # mods: [batch_size, num_modules, out_dim]
                mods = torch.stack(module_outputs, dim=1)
                # Broadcast-multiply and sum over M: y = (attn.unsqueeze(-1) * mods).sum(dim=1)
                weighted_output = (attention.unsqueeze(-1) * mods).sum(dim=1)  # [batch_size, out_dim]
            else:  # conv
                # mods: [batch_size, num_modules, out_channels, H, W]
                mods = torch.stack(module_outputs, dim=1)
                # Broadcast-multiply and sum over M: y = (attn.view(B,M,1,1,1) * mods).sum(dim=1)
                attn_expanded = attention.view(batch_size, self.num_modules, 1, 1, 1)
                weighted_output = (attn_expanded * mods).sum(dim=1)  # [batch_size, out_channels, H, W]
            
            if return_attn:
                attention_info = {
                    'attention': attention,
                    'logits': logits,
                    'regularizers': regularizers
                }
                return weighted_output, attention_info
            else:
                return weighted_output, None
        else:
            # No routing - return all outputs stacked
            if self.block_type == 'mlp':
                # Stack to [batch_size, num_modules, out_dim]
                stacked_output = torch.stack(module_outputs, dim=1)
            else:  # conv
                # Stack to [batch_size, num_modules, out_channels, H, W]
                stacked_output = torch.stack(module_outputs, dim=1)
            
            return stacked_output, None
    
    def _prune_attention(self, attention: Tensor, threshold: float) -> Tensor:
        """Zero-out modules whose attention weight is below threshold.
        
        This is a speed optimization that can be used during inference.
        The math remains correct even with pruning since we renormalize.
        
        Args:
            attention: Attention weights, shape [batch_size, num_modules]
            threshold: Pruning threshold (weights below this are zeroed)
            
        Returns:
            Pruned and renormalized attention weights
        """
        # Create mask for weights above threshold
        mask = attention > threshold
        
        # Zero out weights below threshold
        pruned_attention = attention * mask.float()
        
        # Renormalize to maintain sum = 1
        attention_sum = pruned_attention.sum(dim=-1, keepdim=True)
        # Avoid division by zero
        attention_sum = torch.clamp(attention_sum, min=1e-8)
        pruned_attention = pruned_attention / attention_sum
        
        return pruned_attention
    
    def get_module_outputs(self, x: Tensor) -> List[Tensor]:
        """Get individual module outputs without stacking.
        
        This method is useful for debugging and analysis of individual
        module behaviors.
        
        Args:
            x: Input tensor
            
        Returns:
            List of tensors, one for each module output
        """
        module_outputs = []
        for module in self.modules_list:
            output = module(x)
            module_outputs.append(output)
        return module_outputs
    
    def get_total_parameters(self) -> int:
        """Get total number of parameters across all modules."""
        return sum(p.numel() for p in self.parameters())
    
    def get_parameters_per_module(self) -> int:
        """Get number of parameters per module (should be equal for all modules)."""
        if len(self.modules_list) == 0:
            return 0
        return sum(p.numel() for p in self.modules_list[0].parameters())
    
    def list_init_layers(self) -> List[nn.Module]:
        """Return list of modules whose parameters could be initialized differently.
        
        This method is compatible with the existing continual learning framework
        and returns all linear/conv layers for custom initialization.
        """
        init_layers = []
        for module in self.modules_list:
            if self.block_type == 'mlp':
                init_layers.extend([module.linear1, module.linear2])
            else:  # conv
                init_layers.append(module.conv)
        
        # Add router layers if present
        if self.router is not None:
            init_layers.extend([self.router.linear1, self.router.linear2])
        
        return init_layers
    
    def get_router_regularizers(self, x: Tensor) -> Dict[str, Tensor]:
        """Get router regularization losses for the given input.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary of regularization losses
        """
        if self.router is None:
            return {}
        
        _, _, regularizers = self.router(x)
        return regularizers
    
    def reset_router_ema(self):
        """Reset the router's EMA attention buffer."""
        if self.router is not None:
            self.router.reset_ema()
    
    def get_router_stats(self) -> Dict[str, float]:
        """Get router attention statistics.
        
        Returns:
            Dictionary with router statistics
        """
        if self.router is None:
            return {}
        return self.router.get_attention_stats()
    
    def log_attention_stats(self, attention: Tensor, step: int, task_id: Optional[int] = None, 
                           writer=None, prefix: str = "modular_layer"):
        """Log attention statistics to TensorBoard.
        
        Args:
            attention: Attention weights, shape [batch_size, num_modules]
            step: Current training step
            task_id: Optional task ID for per-task logging
            writer: TensorBoard SummaryWriter
            prefix: Prefix for log names
        """
        if writer is None:
            return
        
        # Log attention histogram
        writer.add_histogram(f"{prefix}/attention_histogram", attention, step)
        
        # Log per-module attention averages
        for i in range(attention.size(1)):
            writer.add_scalar(f"{prefix}/module_{i}_attention", attention[:, i].mean(), step)
        
        # Log attention entropy (measure of sparsity)
        entropy = -torch.sum(attention * torch.log(attention + 1e-8), dim=-1)
        writer.add_scalar(f"{prefix}/attention_entropy", entropy.mean(), step)
        
        # Log attention sparsity (fraction of weights > 0.1)
        sparsity = (attention > 0.1).float().mean()
        writer.add_scalar(f"{prefix}/attention_sparsity", sparsity, step)
        
        # Per-task logging if task_id is provided
        if task_id is not None:
            writer.add_histogram(f"{prefix}/task_{task_id}/attention_histogram", attention, step)
            writer.add_scalar(f"{prefix}/task_{task_id}/attention_entropy", entropy.mean(), step)
            writer.add_scalar(f"{prefix}/task_{task_id}/attention_sparsity", sparsity, step)


# Example usage and testing functions
def create_mlp_modular_layer(in_dim: int, out_dim: int, num_modules: int = 4, 
                           use_router: bool = False, **router_kwargs) -> ModularLayer:
    """Convenience function to create an MLP modular layer."""
    config = ModularLayerConfig(
        in_dim=in_dim,
        out_dim=out_dim,
        num_modules=num_modules,
        block_type='mlp',
        use_router=use_router,
        **router_kwargs
    )
    return ModularLayer(config)


def create_conv_modular_layer(in_channels: int, out_channels: int, num_modules: int = 4,
                            use_router: bool = False, **router_kwargs) -> ModularLayer:
    """Convenience function to create a convolutional modular layer."""
    config = ModularLayerConfig(
        in_dim=in_channels,
        out_dim=out_channels,
        num_modules=num_modules,
        block_type='conv',
        use_router=use_router,
        **router_kwargs
    )
    return ModularLayer(config)
