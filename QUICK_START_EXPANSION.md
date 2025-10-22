# Quick Start: Module Expansion

## 30-Second Overview

**Module Expansion** automatically adds new modules when the router has low confidence, enabling task-free continual learning without knowing task boundaries.

## Minimal Example

```python
from models.modular_layer import ModularLayer, ModularLayerConfig
import torch.optim as optim

# 1. Create layer with expansion enabled
config = ModularLayerConfig(
    in_dim=784,
    out_dim=400,
    num_modules=2,              # Start with 2
    max_modules=8,              # Grow up to 8
    block_type='mlp',
    use_router=True,
    enable_expansion=True,      # Enable expansion
    confidence_threshold=0.4,   # Trigger threshold
    cooldown_steps=500          # Wait between expansions
)
layer = ModularLayer(config)

# 2. Create optimizer
optimizer = optim.Adam(layer.parameters(), lr=0.001)

# 3. In training loop
for batch_idx, (x, y) in enumerate(train_loader):
    # Forward
    output, attn_info = layer(x, return_attn=True)
    loss = criterion(output, y)
    
    # Backward
    loss.backward()
    optimizer.step()
    
    # Check expansion every 10 batches
    if batch_idx % 10 == 0:
        result = layer.maybe_expand(
            attn_info['attention'], 
            optimizer=optimizer
        )
        
        if result['expanded']:
            print(f"Expanded to {layer.num_modules} modules!")
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_expansion` | `False` | Enable dynamic expansion |
| `max_modules` | `8` | Maximum modules allowed |
| `confidence_threshold` | `0.4` | Trigger threshold (lower = harder) |
| `confidence_method` | `'max_weight'` | `'max_weight'` or `'entropy'` |
| `cooldown_steps` | `1000` | Min steps between expansions |
| `projection_steps` | `100` | Projection phase steps |

## Recommended Settings

### SplitMNIST (5 tasks)
```python
num_modules=2
max_modules=6
confidence_threshold=0.4
cooldown_steps=500
```

### SplitCIFAR (10 tasks)
```python
num_modules=3
max_modules=10
confidence_threshold=0.45
cooldown_steps=1000
```

## What Happens During Expansion?

1. **Detect**: Low confidence (< threshold)
2. **Add**: Create new module (initialized from EMA of old ones)
3. **Expand**: Router output M → M+1
4. **Update**: Add new parameters to optimizer
5. **Align** (optional): Run projection phase

## Monitoring

```python
# Get expansion statistics
stats = layer.get_expansion_stats()
print(f"Modules: {stats['num_modules']}/{stats['max_modules']}")
print(f"Expansions: {stats['expansion_count']}")

# Check last expansion result
result = layer.maybe_expand(attention, optimizer)
print(result)
# {'expanded': True/False, 'confidence': 0.35, 
#  'num_modules': 4, 'reason': 'Low confidence...'}
```

## Running the Example

```bash
# Run comprehensive tests
python test_module_expansion.py

# Run full SplitMNIST demonstration
python example_module_expansion.py
```

## Troubleshooting

**No expansions?**
- Increase `confidence_threshold`
- Reduce `cooldown_steps`
- Check `result['reason']` for details

**Too many expansions?**
- Decrease `confidence_threshold`
- Increase `cooldown_steps`
- Increase `router_entropy_coef`

**Training unstable?**
- Enable projection phase: `projection_steps=100`
- Reduce learning rate after expansion
- Increase `router_drift_coef`

## Full Documentation

See [MODULE_EXPANSION_README.md](Documentation/MODULE_EXPANSION_README.md) for:
- Complete API reference
- Advanced usage patterns
- Best practices
- Experimental results
- DDP/distributed training

## Next Steps

1. ✅ Read this quick start
2. 📖 Review [MODULE_EXPANSION_README.md](Documentation/MODULE_EXPANSION_README.md)
3. 🧪 Run `python test_module_expansion.py`
4. 🚀 Run `python example_module_expansion.py`
5. 🔬 Try on your own tasks!

