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
import torch.nn.functional as F
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
        
        # Module Expansion parameters
        enable_expansion: Whether to enable dynamic module expansion
        max_modules: Maximum number of modules allowed (expansion budget)
        confidence_threshold: Threshold for triggering expansion (e.g., 0.4)
        confidence_method: Method for computing confidence ('max_weight' or 'entropy')
        cooldown_steps: Minimum steps between expansions
        projection_steps: Number of projection/distillation steps for new module
        projection_lr: Learning rate for projection phase
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
    
    # Module expansion parameters
    enable_expansion: bool = False
    max_modules: int = 8
    confidence_threshold: float = 0.4
    confidence_method: Literal['max_weight', 'entropy'] = 'max_weight'
    cooldown_steps: int = 1000
    projection_steps: int = 100
    projection_lr: float = 0.001
    
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
        
        # Validate expansion parameters
        if self.enable_expansion:
            if self.max_modules <= self.num_modules:
                raise ValueError("max_modules must be greater than num_modules")
            if not (0.0 < self.confidence_threshold < 1.0):
                raise ValueError("confidence_threshold must be in range (0.0, 1.0)")
            if self.confidence_method not in ['max_weight', 'entropy']:
                raise ValueError("confidence_method must be 'max_weight' or 'entropy'")
            if self.cooldown_steps <= 0:
                raise ValueError("cooldown_steps must be a positive integer")
            if self.projection_steps < 0:
                raise ValueError("projection_steps must be non-negative")
            if not self.use_router:
                raise ValueError("enable_expansion requires use_router=True")


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
        
        # Module expansion tracking
        self.enable_expansion = config.enable_expansion
        self.max_modules = config.max_modules
        self.confidence_threshold = config.confidence_threshold
        self.confidence_method = config.confidence_method
        self.cooldown_steps = config.cooldown_steps
        self.projection_steps = config.projection_steps
        self.projection_lr = config.projection_lr
        
        # Track expansion state
        self.current_step = 0
        self.last_expansion_step = -config.cooldown_steps  # Allow immediate first expansion
        self.expansion_count = 0
        self.expansion_history = []  # List of (step, num_modules, confidence) tuples
        
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
    
    def _compute_confidence(self, attention: Tensor) -> Tensor:
        """Compute confidence score from attention weights.
        
        Confidence indicates how "sure" the router is about module selection.
        Low confidence means the router is spreading weight thinly across modules,
        suggesting that none of the existing modules fit the input well.
        
        Args:
            attention: Attention weights, shape [batch_size, num_modules]
            
        Returns:
            Confidence scores per sample, shape [batch_size]
        """
        if self.confidence_method == 'max_weight':
            # Confidence = max attention weight
            # High max weight = confident selection
            # Low max weight = uncertain, spread thin
            confidence = attention.max(dim=-1)[0]  # [batch_size]
        
        elif self.confidence_method == 'entropy':
            # Confidence = 1 - normalized_entropy
            # Low entropy = confident (peaked distribution)
            # High entropy = uncertain (uniform distribution)
            entropy = -torch.sum(attention * torch.log(attention + 1e-8), dim=-1)  # [batch_size]
            max_entropy = math.log(self.num_modules)  # log(M) for uniform distribution
            normalized_entropy = entropy / (max_entropy + 1e-8)  # [0, 1]
            confidence = 1.0 - normalized_entropy  # [batch_size]
        
        else:
            raise ValueError(f"Unknown confidence_method: {self.confidence_method}")
        
        return confidence
    
    def maybe_expand(self, attention: Tensor, optimizer: Optional[torch.optim.Optimizer] = None,
                    force: bool = False, verbose: bool = True) -> Dict[str, any]:
        """Check if expansion should occur and perform it if conditions are met.
        
        This is the main entry point for expand-on-demand functionality.
        Call this method after forward passes during training to check if
        a new module should be added.
        
        Args:
            attention: Current attention weights, shape [batch_size, num_modules]
            optimizer: Optional optimizer to update with new parameters
            force: Force expansion regardless of conditions (for testing)
            verbose: Whether to print expansion logs
            
        Returns:
            Dictionary with expansion info:
            - 'expanded': Whether expansion occurred
            - 'confidence': Batch mean confidence
            - 'num_modules': Current number of modules
            - 'reason': Reason for expansion or no expansion
        """
        result = {
            'expanded': False,
            'confidence': 0.0,
            'num_modules': self.num_modules,
            'reason': ''
        }
        
        # Increment step counter
        self.current_step += 1
        
        # Compute batch confidence
        confidence = self._compute_confidence(attention)
        batch_confidence = confidence.mean().item()
        result['confidence'] = batch_confidence
        
        # Check expansion conditions
        should_expand, reason = self._check_expansion_conditions(
            batch_confidence, force=force
        )
        result['reason'] = reason
        
        if should_expand:
            # Perform expansion
            if verbose:
                print(f"\n{'='*60}")
                print(f"MODULE EXPANSION TRIGGERED")
                print(f"Step: {self.current_step}")
                print(f"Confidence: {batch_confidence:.4f} (threshold: {self.confidence_threshold:.4f})")
                print(f"Current modules: {self.num_modules} -> {self.num_modules + 1}")
                print(f"Reason: {reason}")
                print(f"[Expansion] Layer adding new module at step {self.current_step}, total modules now {self.num_modules + 1}")
                print(f"{'='*60}\n")
            
            # Add new module
            self._add_new_module()
            
            # Expand router
            self._expand_router()
            
            # Update tracking
            self.expansion_count += 1
            self.last_expansion_step = self.current_step
            self.expansion_history.append((
                self.current_step,
                self.num_modules,
                batch_confidence
            ))
            
            result['expanded'] = True
            result['num_modules'] = self.num_modules
            
            # Update optimizer if provided
            if optimizer is not None:
                self._update_optimizer_params(optimizer)
            
            if verbose:
                print(f"Expansion complete! New module count: {self.num_modules}")
                if self.projection_steps > 0:
                    print(f"Note: Run projection phase with {self.projection_steps} steps")
        
        return result
    
    def _check_expansion_conditions(self, batch_confidence: float, 
                                   force: bool = False) -> Tuple[bool, str]:
        """Check if all conditions for expansion are met.
        
        Args:
            batch_confidence: Mean confidence across the batch
            force: Force expansion regardless of conditions
            
        Returns:
            Tuple of (should_expand, reason)
        """
        if force:
            return True, "Forced expansion"
        
        if not self.enable_expansion:
            return False, "Expansion disabled"
        
        # Check module budget
        if self.num_modules >= self.max_modules:
            return False, f"At max modules ({self.max_modules})"
        
        # Check cooldown period
        steps_since_last = self.current_step - self.last_expansion_step
        if steps_since_last < self.cooldown_steps:
            return False, f"In cooldown ({steps_since_last}/{self.cooldown_steps} steps)"
        
        # Check confidence threshold
        if batch_confidence >= self.confidence_threshold:
            return False, f"Confidence too high ({batch_confidence:.4f} >= {self.confidence_threshold:.4f})"
        
        # All conditions met!
        return True, f"Low confidence ({batch_confidence:.4f} < {self.confidence_threshold:.4f})"
    
    def _add_new_module(self):
        """Add a new module initialized from EMA of existing modules.
        
        The new module is initialized as a weighted average (EMA) of existing
        modules' parameters. This provides a warm start that matches the current
        representation space.
        """
        # Create new module with same architecture
        if self.block_type == 'mlp':
            new_module = MLPBlock(
                in_dim=self.config.in_dim,
                hidden_dim=self.config.hidden_dim,
                out_dim=self.config.out_dim,
                bias=self.config.bias,
                dropout=self.config.dropout
            )
        elif self.block_type == 'conv':
            new_module = ConvBlock(
                in_channels=self.config.in_dim,
                out_channels=self.config.out_dim,
                bias=self.config.bias,
                dropout=self.config.dropout
            )
        else:
            raise ValueError(f"Unknown block_type: {self.block_type}")
        
        # Initialize from EMA of existing modules
        self._initialize_module_from_ema(new_module)
        
        # Add to module list
        new_module.to(self.config.device)
        self.modules_list.append(new_module)
        self.num_modules += 1
    
    def _initialize_module_from_ema(self, new_module: nn.Module, ema_weight: float = 0.1):
        """Initialize new module as weighted average of existing modules.
        
        Args:
            new_module: The new module to initialize
            ema_weight: Weight for averaging (lower = closer to mean)
        """
        with torch.no_grad():
            # Get parameters from all existing modules
            for param_name, new_param in new_module.named_parameters():
                # Compute mean of corresponding parameters from existing modules
                param_sum = None
                count = 0
                
                for module in self.modules_list:
                    for name, param in module.named_parameters():
                        if name == param_name:
                            if param_sum is None:
                                param_sum = param.data.clone()
                            else:
                                param_sum += param.data
                            count += 1
                            break
                
                if param_sum is not None and count > 0:
                    # Set new parameter to mean of existing
                    param_mean = param_sum / count
                    # Add small noise for diversity
                    noise = torch.randn_like(param_mean) * ema_weight * param_mean.std()
                    new_param.data.copy_(param_mean + noise)
    
    def _expand_router(self):
        """Expand router output dimension from M to M+1.
        
        This adds a new output logit for the new module, initialized with
        low bias so it doesn't immediately dominate routing.
        """
        if self.router is None:
            return
        
        # Get current router output layer
        old_linear2 = self.router.linear2
        old_out_features = old_linear2.out_features
        in_features = old_linear2.in_features
        
        # Create new output layer with M+1 outputs
        new_linear2 = nn.Linear(in_features, old_out_features + 1, 
                               bias=old_linear2.bias is not None)
        new_linear2.to(self.config.device)
        
        # Copy old weights and biases
        with torch.no_grad():
            # Copy existing weights
            new_linear2.weight.data[:old_out_features, :] = old_linear2.weight.data
            
            # Initialize new module's logit weight as mean of existing
            mean_weight = old_linear2.weight.data.mean(dim=0, keepdim=True)
            new_linear2.weight.data[old_out_features:, :] = mean_weight * 0.1  # Small scale
            
            if old_linear2.bias is not None:
                # Copy existing biases
                new_linear2.bias.data[:old_out_features] = old_linear2.bias.data
                # Initialize new bias low (negative) so new module starts with low probability
                new_linear2.bias.data[old_out_features:] = -2.0
        
        # Replace router output layer
        self.router.linear2 = new_linear2
        self.router.num_modules = self.num_modules
        
        # Expand EMA buffer
        old_ema = self.router.ema_attention
        new_ema = torch.zeros(1, self.num_modules, device=self.config.device)
        with torch.no_grad():
            new_ema[0, :old_out_features] = old_ema[0, :]
            new_ema[0, old_out_features] = 1.0 / self.num_modules  # Small initial value
            # Renormalize
            new_ema = new_ema / new_ema.sum()
        self.router.ema_attention = new_ema
    
    def run_projection_phase(self, x: Tensor, num_steps: Optional[int] = None,
                            lr: Optional[float] = None, verbose: bool = True) -> List[float]:
        """Run projection/distillation phase for the newest module.
        
        After adding a new module, run this to train it to match the output
        of the weighted combination of old modules. This aligns it with the
        existing representation space.
        
        Args:
            x: Training data, shape [batch_size, in_dim] or [batch_size, in_channels, H, W]
            num_steps: Number of projection steps (default: self.projection_steps)
            lr: Learning rate (default: self.projection_lr)
            verbose: Whether to print progress
            
        Returns:
            List of projection losses over steps
        """
        if self.num_modules <= 1:
            if verbose:
                print("No projection needed (only 1 module)")
            return []
        
        num_steps = num_steps or self.projection_steps
        lr = lr or self.projection_lr
        
        if num_steps == 0:
            return []
        
        if verbose:
            print(f"\nRunning projection phase: {num_steps} steps, lr={lr}")
        
        # Get the newest module
        newest_module = self.modules_list[-1]
        
        # Create optimizer for only the new module
        optimizer = torch.optim.Adam(newest_module.parameters(), lr=lr)
        
        losses = []
        
        for step in range(num_steps):
            # Compute outputs from all modules
            module_outputs = []
            for module in self.modules_list:
                with torch.no_grad() if module != newest_module else torch.enable_grad():
                    output = module(x)
                    module_outputs.append(output)
            
            # Get router attention (excluding new module)
            if self.block_type == 'conv':
                batch_size, channels, height, width = x.shape
                x_flat = x.view(batch_size, -1)
                _, attention, _ = self.router(x_flat)
            else:
                _, attention, _ = self.router(x)
            
            # Compute target: weighted sum of OLD modules (exclude newest)
            with torch.no_grad():
                if self.block_type == 'mlp':
                    old_mods = torch.stack(module_outputs[:-1], dim=1)  # [B, M-1, out_dim]
                    old_attn = attention[:, :-1]  # [B, M-1]
                    # Renormalize old attention
                    old_attn = old_attn / (old_attn.sum(dim=-1, keepdim=True) + 1e-8)
                    target = (old_attn.unsqueeze(-1) * old_mods).sum(dim=1)  # [B, out_dim]
                else:  # conv
                    old_mods = torch.stack(module_outputs[:-1], dim=1)  # [B, M-1, C, H, W]
                    old_attn = attention[:, :-1]  # [B, M-1]
                    old_attn = old_attn / (old_attn.sum(dim=-1, keepdim=True) + 1e-8)
                    old_attn_exp = old_attn.view(x.size(0), self.num_modules - 1, 1, 1, 1)
                    target = (old_attn_exp * old_mods).sum(dim=1)  # [B, C, H, W]
            
            # Get new module output
            new_output = module_outputs[-1]
            
            # Compute MSE loss
            loss = F.mse_loss(new_output, target)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if verbose and (step % (num_steps // 10) == 0 or step == num_steps - 1):
                print(f"  Step {step}/{num_steps}: loss = {loss.item():.6f}")
        
        if verbose:
            print(f"Projection phase complete. Final loss: {losses[-1]:.6f}\n")
        
        return losses
    
    def get_new_optimizer_params(self) -> List[Dict]:
        """Get parameter groups for newly added module and router weights.
        
        Use this to update optimizer after expansion:
            new_params = layer.get_new_optimizer_params()
            for param_group in new_params:
                optimizer.add_param_group(param_group)
        
        Returns:
            List of parameter group dictionaries
        """
        param_groups = []
        
        # Add newest module parameters
        if len(self.modules_list) > 0:
            newest_module = self.modules_list[-1]
            param_groups.append({
                'params': list(newest_module.parameters()),
                'name': f'module_{self.num_modules - 1}'
            })
        
        # Add new router parameters (the expanded weights)
        if self.router is not None:
            # Note: Only the new weights/biases in linear2 are actually new
            # For simplicity, we add the whole linear2 layer
            param_groups.append({
                'params': list(self.router.linear2.parameters()),
                'name': 'router_output'
            })
        
        return param_groups
    
    def _update_optimizer_params(self, optimizer: torch.optim.Optimizer):
        """Update optimizer with new module parameters.
        
        Args:
            optimizer: The optimizer to update
        """
        new_param_groups = self.get_new_optimizer_params()
        for param_group in new_param_groups:
            optimizer.add_param_group(param_group)
    
    def get_expansion_stats(self) -> Dict[str, any]:
        """Get statistics about module expansions.
        
        Returns:
            Dictionary with expansion statistics
        """
        return {
            'num_modules': self.num_modules,
            'expansion_count': self.expansion_count,
            'current_step': self.current_step,
            'last_expansion_step': self.last_expansion_step,
            'expansion_history': self.expansion_history.copy(),
            'enable_expansion': self.enable_expansion,
            'max_modules': self.max_modules,
            'confidence_threshold': self.confidence_threshold,
        }
    
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
