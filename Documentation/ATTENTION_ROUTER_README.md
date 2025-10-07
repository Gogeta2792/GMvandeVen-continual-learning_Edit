# AttentionRouter Implementation

This implementation provides a task-agnostic attention router for modular continual learning, designed for the FYP project "Modular Continual Learning with Task-Agnostic Subnetwork Routing via Attention Mechanisms."

## Overview

The `AttentionRouter` class computes attention weights over M modules based on input characteristics, without requiring task labels. It supports both local and global routing modes with regularization for stable attention patterns to prevent abrupt routing changes.

## Key Components

### 1. RouterConfig
A dataclass that configures the attention router with parameters like:
- `num_modules`: Number of modules to route between (M)
- `input_dim`: Input dimension for local routing
- `global_dim`: Global context dimension (0 if not using global routing)
- `hidden_dim`: Hidden dimension in the router MLP
- `use_global`: Whether to use global routing mode
- `entropy_coef`: Coefficient for entropy regularization (encourage sparsity)
- `drift_coef`: Coefficient for L2 drift regularization (encourage stability)

### 2. AttentionRouter
Main router class with architecture: `LayerNorm → Linear → GELU → Linear → logits`
- **Local routing**: Takes current layer's input and outputs attention logits
- **Global routing**: Optionally takes global context vector concatenated with input
- **Regularization**: Entropy penalty and L2 drift penalty for stable attention

### 3. LocalAttentionRouter & GlobalAttentionRouter
Convenience classes for local-only and global routing respectively.

## Usage Examples

### Basic Local Router
```python
from models.attention_router import create_local_router

# Create a local router for 4 modules
router = create_local_router(
    num_modules=4,
    input_dim=100,
    hidden_dim=128,
    entropy_coef=0.01,  # Encourage sparse attention
    drift_coef=0.1      # Encourage stable attention
)

# Forward pass
x = torch.randn(32, 100)
logits, attention, regularizers = router(x)

print(f"Logits shape: {logits.shape}")      # [32, 4]
print(f"Attention shape: {attention.shape}") # [32, 4]
print(f"Regularizers: {list(regularizers.keys())}")  # ['entropy_penalty', 'drift_penalty']
```

### Global Router
```python
from models.attention_router import create_global_router

# Create a global router
router = create_global_router(
    num_modules=6,
    input_dim=100,
    global_dim=50,
    hidden_dim=128
)

# Forward pass with global context
x = torch.randn(32, 100)
global_context = torch.randn(32, 50)
logits, attention, regularizers = router(x, global_context)
```

### Integration with ModularLayer
```python
from models.modular_layer import create_mlp_modular_layer

# Create modular layer with router
layer = create_mlp_modular_layer(
    in_dim=784,      # MNIST input
    out_dim=400,     # Hidden size
    num_modules=4,   # 4 parallel modules
    use_router=True, # Enable attention routing
    router_hidden_dim=128,
    router_entropy_coef=0.01,
    router_drift_coef=0.1
)

# Forward pass
x = torch.randn(32, 784)
output, attention_info = layer(x, return_attention=True)

print(f"Output shape: {output.shape}")  # [32, 400]
print(f"Attention shape: {attention_info['attention'].shape}")  # [32, 4]
print(f"Regularizers: {list(attention_info['regularizers'].keys())}")
```

## Router Regularization

The router includes two key regularizers for stable attention:

### 1. Entropy Penalty
Encourages confident, sparse module selections:
```python
entropy = -torch.sum(attention * torch.log(attention + 1e-8), dim=-1)
entropy_penalty = entropy_coef * entropy.mean()
```

### 2. L2 Drift Penalty
Discourages abrupt routing changes using EMA:
```python
current_mean = attention.mean(dim=0, keepdim=True)
drift = F.mse_loss(current_mean, ema_attention)
drift_penalty = drift_coef * drift
```

## Training Integration

The router regularizers can be easily integrated into training loops:

```python
class ModularNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.modular_layer = create_mlp_modular_layer(
            in_dim=784, out_dim=400, num_modules=4, use_router=True
        )
        self.classifier = nn.Linear(400, 10)
    
    def forward(self, x, return_regularizers=False):
        x = x.view(x.size(0), -1)
        x, attn_info = self.modular_layer(x, return_attention=True)
        output = self.classifier(x)
        
        if return_regularizers:
            return output, attn_info['regularizers']
        else:
            return output

# Training loop
network = ModularNetwork()
optimizer = optim.Adam(network.parameters(), lr=0.001)

for batch in dataloader:
    x, y = batch
    output, regularizers = network(x, return_regularizers=True)
    
    # Compute losses
    ce_loss = nn.CrossEntropyLoss()(output, y)
    reg_loss = sum(regularizers.values())
    total_loss = ce_loss + reg_loss
    
    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
```

## Router Statistics and Analysis

The router provides methods for analyzing attention patterns:

```python
# Get router statistics
stats = router.get_attention_stats()
print(f"EMA attention: {stats['ema_attention']}")
print(f"EMA entropy: {stats['ema_entropy']}")

# Reset EMA buffer
router.reset_ema()

# Analyze attention patterns
x = torch.randn(100, 100)
logits, attention, regularizers = router(x)

# Module usage analysis
module_usage = (attention > 0.1).float().mean(dim=0)
print(f"Module usage: {module_usage}")

# Entropy analysis
entropy = -torch.sum(attention * torch.log(attention + 1e-8), dim=-1)
print(f"Mean entropy: {entropy.mean()}")
```

## Configuration Options

### Router Flags
- `use_global=False`: Enable global routing mode
- `entropy_coef=0.01`: Entropy regularization coefficient
- `drift_coef=0.1`: Drift regularization coefficient
- `hidden_dim=128`: Hidden dimension in router MLP
- `ema_momentum=0.99`: EMA momentum for drift regularization

### Regularization Effects
- **No regularization**: Uniform attention, high entropy
- **Entropy only**: Sparse attention, confident selections
- **Drift only**: Stable attention, prevents abrupt changes
- **Both**: Balanced sparse and stable attention

## Integration with Existing Framework

The router is designed to integrate seamlessly with the existing continual learning framework:

```python
# Compatible with existing initialization
init_layers = layer.list_init_layers()  # Includes router layers

# Compatible with existing parameter counting
total_params = sum(p.numel() for p in layer.parameters())

# Router-specific methods
regularizers = layer.get_router_regularizers(x)
stats = layer.get_router_stats()
layer.reset_router_ema()
```

## Files

- `models/attention_router.py`: Main router implementation
- `example_attention_router.py`: Comprehensive usage examples
- `Documentation/ATTENTION_ROUTER_README.md`: This documentation

## Testing

The implementation includes comprehensive unit tests that verify:
- ✅ Correct output shapes for different batch sizes
- ✅ Attention weights sum to 1 for each sample
- ✅ Gradient flow through all router components
- ✅ Regularizer computation and integration
- ✅ EMA updates and statistics
- ✅ Performance (CPU-only, <2s)

## Key Features

1. **Task-Agnostic**: No task labels required for routing
2. **Input-Driven**: Routing based on input characteristics (like LMC)
3. **Stable Attention**: Regularization prevents abrupt routing changes
4. **Flexible**: Supports both local and global routing modes
5. **Integrated**: Seamlessly works with ModularLayer
6. **Analyzable**: Provides attention statistics and patterns

## Next Steps for FYP

1. **Integrate** AttentionRouter into SplitMNIST experiments
2. **Experiment** with different regularization coefficients
3. **Analyze** attention patterns across different tasks
4. **Compare** with baseline methods (SI/EWC/ER)
5. **Extend** to more complex scenarios (CIFAR-10/100)

The AttentionRouter provides a solid foundation for your modular continual learning research with task-agnostic subnetwork routing!
