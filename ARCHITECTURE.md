# Architecture

This document describes the technical architecture of the Modular Continual Learning system with Task-Agnostic Subnetwork Routing via Attention Mechanisms.

## Overview

The system implements a modular neural network architecture that can dynamically route information through subnetworks using attention mechanisms, enabling task-free continual learning without explicit task boundaries.

## Core Components

### 1. ModularLayer

The fundamental building block of the system, implementing a modular layer with dynamic routing capabilities.

#### Configuration (`ModularLayerConfig`)
```python
class ModularLayerConfig:
    in_dim: int                    # Input dimension
    out_dim: int                   # Output dimension
    num_modules: int               # Initial number of modules
    max_modules: int               # Maximum modules (expansion budget)
    block_type: str                # 'mlp' or 'conv'
    use_router: bool               # Enable attention routing
    enable_expansion: bool         # Enable dynamic expansion
    confidence_threshold: float    # Expansion trigger threshold
    confidence_method: str         # 'max_weight' or 'entropy'
    cooldown_steps: int           # Steps between expansions
    projection_steps: int         # Projection phase length
```

#### Key Methods
- `forward(x, return_attn=False)`: Forward pass with optional attention return
- `maybe_expand(attention, optimizer)`: Check and perform expansion
- `run_projection_phase(x, num_steps)`: Align new module with existing output
- `get_expansion_stats()`: Get expansion statistics and history

### 2. Attention Router

Implements attention-based routing to select and weight modules dynamically.

#### LocalAttentionRouter
- Learns attention weights for each module
- Supports entropy and drift regularization
- Maintains exponential moving average of attention patterns
- Handles router expansion when new modules are added

#### Router Types
1. **Attention Router**: Learned routing with attention mechanisms
2. **Null Router**: Fixed deterministic routing (selects module 0)

### 3. Module Expansion System

Dynamic capacity growth system that automatically adds modules when needed.

#### Expansion Trigger
The system monitors router confidence and triggers expansion when:
- Confidence < threshold (indicating no module fits current input)
- Under module budget (< max_modules)
- Cooldown period satisfied
- Expansion enabled

#### Confidence Computation
Two methods implemented:

**Max-Weight Method (Default)**:
```python
confidence = max(attention_weights)
# High max = confident, low max = uncertain
```

**Entropy Method**:
```python
entropy = -sum(p * log(p))
normalized_entropy = entropy / log(num_modules)
confidence = 1 - normalized_entropy
# Low entropy = confident, high entropy = uncertain
```

#### Module Addition Process
1. **Detect**: Low confidence (< threshold)
2. **Add**: Create new module initialized from EMA of existing modules
3. **Expand**: Router output M → M+1
4. **Update**: Add new parameters to optimizer
5. **Align**: Optional projection phase

#### Smart Initialization
New modules are initialized as:
```python
param_mean = sum(module.params) / num_modules
new_param = param_mean + noise * 0.1 * std(param_mean)
```

This provides a warm start in the existing representation space.

#### Router Expansion
New router logit initialized with:
- **Small weights**: `mean_old_weights * 0.1`
- **Negative bias**: `-2.0`

Prevents new module from immediately dominating routing.

### 4. Balanced Training System

Comprehensive training schedule that alternates emphasis between router and modules to prevent routing collapse.

#### Loss Components
```python
loss_cls = F.cross_entropy(output, target)                    # Classification loss
loss_router = entropy_coef*H(attn) + drift_coef*||attn-ema||^2  # Router regularization
loss_sparsity = sparsity_coef * L1(mean_attention)           # Optional sparsity loss
total_loss = loss_cls + loss_router + loss_sparsity
```

#### Two-Optimizer System
```python
opt_modules = optim.Adam(model.get_module_parameters(), lr=0.001)    # Higher LR
opt_router = optim.Adam(model.get_router_parameters(), lr=0.0001)    # Lower LR
```

#### Curriculum Learning
```python
if step_count < freeze_router_steps:
    model.freeze_router(freeze=True)
else:
    model.freeze_router(freeze=False)
```

#### Early Stopping for Router Collapse
```python
if model.check_router_collapse(attention):
    collapse_steps += 1
    if collapse_steps >= max_collapse_steps:
        router_collapsed = True
        # Stop router updates
```

## Network Architectures

### SimpleModularMLP
```
Input: 784 (28×28 MNIST flattened)
  ↓
Feature Layer (modular or standard):
  - With router: ModularLayer(num_modules, use_router=True)
  - Without router: Standard MLP (Linear → ReLU → Linear)
  ↓ (hidden_dim=400)
Task Heads (multi-head):
  - Head 0: Linear(400, 2) for classes {0, 1}
  - Head 1: Linear(400, 2) for classes {2, 3}
  - Head 2: Linear(400, 2) for classes {4, 5}
  - Head 3: Linear(400, 2) for classes {6, 7}
  - Head 4: Linear(400, 2) for classes {8, 9}
```

### ExpandableModularNetwork
Multi-layer network with expansion capabilities:
- Multiple ModularLayers with independent expansion
- Shared expansion statistics tracking
- Coordinated expansion across layers

## Training Protocols

### Task-Incremental (Multi-Head)
- **5 tasks**: Binary classification on digit pairs
  - Task 0: {0 vs 1}
  - Task 1: {2 vs 3}
  - Task 2: {4 vs 5}
  - Task 3: {6 vs 7}
  - Task 4: {8 vs 9}
- **Multi-head**: Separate classifier head per task
- **Evaluation**: Use correct head for each task

### Task-Free Continual Learning
- No explicit task boundaries
- Continuous data stream
- Dynamic expansion based on confidence
- No task identity required at test time

## Key Design Principles

### 1. Task-Agnostic Learning
- Remove need for explicit task identity at test time
- Dynamic routing through subnetworks
- Attention mechanisms for flexible routing

### 2. Modular Architecture
- Decompose network into reusable building blocks
- Selective module activation
- Knowledge sharing across tasks
- Minimize interference between tasks

### 3. Adaptive Capacity
- Grow only when necessary
- Balance under-capacity vs over-capacity
- Efficient parameter utilization
- Better accuracy with fewer parameters

### 4. Stable Training
- Prevent routing collapse
- Smooth module integration
- Comprehensive safeguards
- Reproducible results

## Configuration Guidelines

### Recommended Settings

#### SplitMNIST (5 tasks)
```python
num_modules=2
max_modules=6
confidence_threshold=0.4
cooldown_steps=500
projection_steps=100
```

#### SplitCIFAR (10 tasks)
```python
num_modules=3
max_modules=10
confidence_threshold=0.45
cooldown_steps=1000
projection_steps=200
```

### Hyperparameter Recommendations
- **Start Conservative**: Use default coefficients first
- **Monitor Attention**: Watch for collapse or uniform attention
- **Adjust Gradually**: Small changes to coefficients
- **Task-Specific**: Different tasks may need different settings

### Common Issues & Solutions
- **Router Collapse**: Increase `entropy_coef`, decrease `lr_router`
- **Uniform Attention**: Decrease `entropy_coef`, increase `sparsity_coef`
- **Unstable Training**: Increase `drift_coef`, decrease learning rates

## Performance Characteristics

### Expected Results
- **>97% accuracy** on SplitMNIST with 5 tasks
- **Stable attention patterns** without collapse
- **Efficient module utilization** (not all modules active)
- **Reproducible results** with proper seeding

### Scalability
- Supports both MLP and convolutional architectures
- DDP compatible for distributed training
- Memory efficient with module sharing
- Linear scaling with number of modules

## Integration Points

### With Existing Continual Learning Framework
- Compatible with existing `--experiment=splitMNIST --scenario=task` pipeline
- Supports standard continual learning methods (SI, EWC, ER, etc.)
- Integrates with existing evaluation metrics
- Works with existing data loading and preprocessing

### CLI Integration
```bash
# All requested CLI flags implemented
--lr_modules 0.001          # Learning rate for modules
--lr_router 0.0001          # Learning rate for router
--entropy_coef 0.01         # Entropy regularization coefficient
--drift_coef 0.1            # Drift regularization coefficient
--sparsity_coef 0.001       # Sparsity regularization coefficient
--freeze_router_steps 500   # Steps to freeze router initially
--collapse_threshold 0.95   # Router collapse threshold
--max_collapse_steps 100    # Max steps before stopping on collapse
```

## Detailed Documentation

For comprehensive implementation details, see:
- `Documentation/MODULE_EXPANSION_README.md` - Complete module expansion documentation
- `Documentation/BALANCED_TRAINING_README.md` - Balanced training system details
- `Documentation/MODULAR_LAYER_README.md` - Modular layer implementation
- `Documentation/ATTENTION_ROUTER_README.md` - Attention router details
- `Documentation/ATTENTION_WEIGHTED_SUM_README.md` - Attention mechanisms

## Future Enhancements

Possible extensions:
1. **Module pruning**: Remove under-utilized modules
2. **Adaptive thresholds**: Learn confidence threshold from data
3. **Task-specific expansion**: Different budgets per layer
4. **Meta-learning initialization**: Learn good initialization strategy
5. **Compression**: Distill expanded network back to smaller size
