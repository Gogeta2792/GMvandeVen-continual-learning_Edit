# ModularLayer Implementation

This implementation provides a modular layer system for continual learning, designed for the FYP project "Modular Continual Learning with Task-Agnostic Subnetwork Routing via Attention Mechanisms."

## Overview

The `ModularLayer` class contains multiple parallel sub-modules that can be selectively activated via attention mechanisms. This enables task-agnostic subnetwork routing and skill reuse across tasks, similar to the LMC (Learnable Modular Components) approach.

## Key Components

### 1. ModularLayerConfig
A dataclass that configures the modular layer with parameters like:
- `in_dim`, `out_dim`: Input/output dimensions
- `num_modules`: Number of parallel sub-modules (M)
- `block_type`: 'mlp' or 'conv'
- `hidden_dim`, `bias`, `dropout`, `batch_norm`: Architecture options

### 2. MLPBlock
Two-layer MLP: `Linear → ReLU → Linear`
- Configurable hidden dimension
- Optional dropout and bias terms
- Designed for fully connected layers

### 3. ConvBlock  
Convolutional block: `Conv3×3 → BN → ReLU`
- 3x3 convolution with padding
- Batch normalization
- Optional dropout
- Designed for image processing (CIFAR)

### 4. ModularLayer
Main class that owns M parallel sub-modules:
- `forward(x)` returns `[B, M, D]` for MLP or `[B, M, C, H, W]` for Conv
- Computes all module outputs (no routing yet)
- Compatible with existing continual learning framework

## Usage Examples

### Basic MLP Layer
```python
from models.modular_layer import ModularLayer, ModularLayerConfig

config = ModularLayerConfig(
    in_dim=784,      # MNIST input
    out_dim=400,     # Hidden size
    num_modules=4,   # 4 parallel modules
    block_type='mlp'
)
layer = ModularLayer(config)

# Forward pass
x = torch.randn(32, 784)
output = layer(x)  # Shape: [32, 4, 400]
```

### Basic Conv Layer
```python
config = ModularLayerConfig(
    in_dim=64,       # Input channels
    out_dim=128,     # Output channels  
    num_modules=6,   # 6 parallel modules
    block_type='conv'
)
layer = ModularLayer(config)

# Forward pass
x = torch.randn(32, 64, 32, 32)
output = layer(x)  # Shape: [32, 6, 128, 32, 32]
```

### Convenience Functions
```python
from models.modular_layer import create_mlp_modular_layer, create_conv_modular_layer

# Quick creation
mlp_layer = create_mlp_modular_layer(in_dim=784, out_dim=400, num_modules=4)
conv_layer = create_conv_modular_layer(in_channels=64, out_channels=128, num_modules=6)
```

## Integration with Existing Framework

The implementation is designed to integrate seamlessly with the existing continual learning framework:

```python
class ModularNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.modular_layers = nn.ModuleList([
            ModularLayer(config1),
            ModularLayer(config2)
        ])
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        for layer in self.modular_layers:
            x = layer(x)  # [B, M, D]
            x = x[:, 0, :]  # For now, use first module (later: attention)
        return self.classifier(x)
    
    def list_init_layers(self):
        """Compatible with existing framework initialization."""
        init_layers = []
        for layer in self.modular_layers:
            init_layers.extend(layer.list_init_layers())
        return init_layers
```

## Testing

Run the comprehensive test suite:
```bash
python test_modular_layer.py
```

Tests verify:
- ✅ Correct parameter counts per module
- ✅ Output shapes for batch sizes 1 and 32
- ✅ Gradient flow through all modules
- ✅ Performance (CPU-only, <2s)
- ✅ Integration with existing framework

## Future: Attention Routing

The current implementation provides all module outputs. Future attention routing will:

1. **Compute attention weights** based on input characteristics
2. **Soft-select modules** using learned attention mechanisms  
3. **Combine outputs** via weighted sum: `Σ(attention_weights * module_outputs)`
4. **Enable task-agnostic routing** without explicit task labels

Example of future attention integration:
```python
# Current: All outputs
all_outputs = layer(x)  # [B, M, D]

# Future: Attention routing
attention_weights = attention_router(x)  # [B, M]
attended_output = torch.sum(all_outputs * attention_weights.unsqueeze(-1), dim=1)
```

## Files

- `models/modular_layer.py`: Main implementation
- `test_modular_layer.py`: Comprehensive unit tests
- `example_modular_layer.py`: Usage examples and integration demos
- `MODULAR_LAYER_README.md`: This documentation

## Next Steps for FYP

1. **Integrate** ModularLayer into existing SplitMNIST pipeline
2. **Implement** attention-based routing mechanism
3. **Train** and compare with baseline methods (SI/EWC/ER)
4. **Extend** to CIFAR-10/100 for more complex scenarios
5. **Analyze** module specialization and skill reuse patterns

The implementation provides a solid foundation for your modular continual learning research!
