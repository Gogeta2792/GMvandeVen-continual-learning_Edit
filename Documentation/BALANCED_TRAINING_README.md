# Balanced Training Schedule for Modular Continual Learning

This implementation provides a comprehensive training loop that alternates emphasis between router and modules to prevent routing collapse, based on the documentation from the AttentionRouter, Attention-Weighted Sum, and ModularLayer implementations.

## Overview

The balanced training system addresses the key challenge in modular continual learning: preventing the attention router from collapsing to a single expert module. It implements a sophisticated training schedule with multiple loss components, two-optimizer system, curriculum learning, and early stopping mechanisms.

## Key Features

### 1. **Balanced Loss Computation**
```python
# Core loss components
loss_cls = F.cross_entropy(output, target)                    # Classification loss
loss_router = entropy_coef*H(attn) + drift_coef*||attn-ema||^2  # Router regularization
loss_sparsity = sparsity_coef * L1(mean_attention)           # Optional sparsity loss
total_loss = loss_cls + loss_router + loss_sparsity
```

**Components:**
- **Classification Loss**: Standard cross-entropy for task performance
- **Router Loss**: Entropy penalty (encourage confident selections) + drift penalty (encourage stability)
- **Sparsity Loss**: L1 penalty on mean attention to keep few modules active

### 2. **Two-Optimizer System**
```python
# Separate optimizers for different components
opt_modules = optim.Adam(model.get_module_parameters(), lr=0.001)    # Higher LR
opt_router = optim.Adam(model.get_router_parameters(), lr=0.0001)    # Lower LR
```

**Benefits:**
- **Modules**: Higher learning rate for fast adaptation to new tasks
- **Router**: Lower learning rate for stable attention patterns
- **Independent Updates**: Prevents router from being overwhelmed by module updates

### 3. **Curriculum Learning**
```python
# Freeze router for first N steps to let modules learn
if step_count < freeze_router_steps:
    model.freeze_router(freeze=True)
else:
    model.freeze_router(freeze=False)
```

**Strategy:**
- **Phase 1**: Freeze router, let modules learn task-specific representations
- **Phase 2**: Unfreeze router, learn attention patterns over trained modules
- **Prevents**: Router from making routing decisions before modules are trained

### 4. **Early Stopping for Router Collapse**
```python
# Detect if router collapses to one module
if model.check_router_collapse(attention):
    collapse_steps += 1
    if collapse_steps >= max_collapse_steps:
        router_collapsed = True
        # Stop router updates
```

**Detection:**
- **Threshold**: If any module gets >95% attention for K consecutive steps
- **Action**: Stop router updates, continue with module training
- **Prevents**: Wasted computation on collapsed routing

## Usage Examples

### Basic Training
```bash
# Run balanced training on SplitMNIST
python train_splitmnist_balanced.py \
    --use_router \
    --gpu \
    --lr_modules 0.001 \
    --lr_router 0.0001 \
    --entropy_coef 0.01 \
    --drift_coef 0.1 \
    --sparsity_coef 0.001 \
    --freeze_router_steps 500
```

### Advanced Configuration
```bash
# Custom configuration for different scenarios
python train_splitmnist_balanced.py \
    --experiment splitMNIST \
    --scenario task \
    --contexts 5 \
    --num_modules 6 \
    --fc_layers 3 \
    --fc_units 512 \
    --lr_modules 0.002 \
    --lr_router 0.0005 \
    --entropy_coef 0.005 \
    --drift_coef 0.05 \
    --sparsity_coef 0.0005 \
    --freeze_router_steps 1000 \
    --collapse_threshold 0.9 \
    --max_collapse_steps 50 \
    --log_dir ./logs/balanced_training
```

### CLI Flags Reference

#### **General Options**
- `--experiment`: Experiment name (default: splitMNIST)
- `--scenario`: Scenario type (default: task)
- `--contexts`: Number of contexts/tasks (default: 5)
- `--seed`: Random seed for reproducibility (default: 42)
- `--gpu`: Use GPU if available

#### **Model Options**
- `--num_modules`: Number of modules per layer (default: 4)
- `--fc_layers`: Number of fully connected layers (default: 2)
- `--fc_units`: Number of units per layer (default: 400)
- `--use_router`: Enable attention routing

#### **Training Options**
- `--lr_modules`: Learning rate for modules (default: 0.001)
- `--lr_router`: Learning rate for router (default: 0.0001)
- `--batch`: Batch size (default: 128)
- `--epochs`: Epochs per task (default: 20)

#### **Loss Coefficients**
- `--entropy_coef`: Entropy regularization coefficient (default: 0.01)
- `--drift_coef`: Drift regularization coefficient (default: 0.1)
- `--sparsity_coef`: Sparsity regularization coefficient (default: 0.001)

#### **Curriculum Learning**
- `--freeze_router_steps`: Steps to freeze router initially (default: 500)

#### **Router Collapse Detection**
- `--collapse_threshold`: Router collapse threshold (default: 0.95)
- `--max_collapse_steps`: Max steps before stopping on collapse (default: 100)

#### **Logging**
- `--log_dir`: TensorBoard log directory
- `--save_dir`: Model save directory

## Training Loop Architecture

### 1. **Initialization Phase**
```python
# Set up two-optimizer system
opt_modules = optim.Adam(model.get_module_parameters(), lr=lr_modules)
opt_router = optim.Adam(model.get_router_parameters(), lr=lr_router)

# Initialize curriculum learning
model.freeze_router(freeze=True)
```

### 2. **Training Step**
```python
def train_step(data, target, task_id):
    # Forward pass
    output, regularizers, attention = model(data, return_regularizers=True, return_attention=True)
    
    # Compute losses
    loss_cls = F.cross_entropy(output, target)
    loss_router = sum(regularizers.values())
    loss_sparsity = model.compute_sparsity_loss(attention)
    total_loss = loss_cls + loss_router + loss_sparsity
    
    # Backward pass
    opt_modules.zero_grad()
    opt_router.zero_grad()
    total_loss.backward()
    
    # Update parameters
    opt_modules.step()
    if not model.router_frozen and not router_collapsed:
        opt_router.step()
    
    # Check for router collapse
    if model.check_router_collapse(attention):
        handle_router_collapse()
    
    return step_stats
```

### 3. **Curriculum Learning**
```python
def handle_curriculum_learning(step_count):
    if step_count < freeze_router_steps and model.router_frozen:
        # Keep router frozen
        pass
    elif step_count >= freeze_router_steps and model.router_frozen:
        # Unfreeze router
        model.freeze_router(freeze=False)
        print(f"Unfreezing router at step {step_count}")
```

### 4. **Router Collapse Handling**
```python
def handle_router_collapse(attention):
    if model.check_router_collapse(attention):
        model.collapse_steps += 1
        if model.collapse_steps >= model.max_collapse_steps:
            router_collapsed = True
            print(f"Router collapsed! Stopping router updates.")
    else:
        model.collapse_steps = 0
```

## Integration with Existing Framework

The balanced training system is designed to integrate seamlessly with the existing continual learning framework:

### **Compatible with Existing Pipeline**
```bash
# Works with existing experiment setup
python train_splitmnist_balanced.py \
    --experiment splitMNIST \
    --scenario task \
    --contexts 5 \
    --d_dir ./data \
    --normalize
```

### **Framework Compatibility**
- ✅ Uses existing `get_context_set()` for data loading
- ✅ Compatible with existing model initialization
- ✅ Supports existing evaluation protocols
- ✅ Works with existing parameter counting methods

### **Extension Points**
```python
# Easy to extend with additional regularizers
class ExtendedBalancedNetwork(BalancedModularNetwork):
    def compute_additional_losses(self, attention):
        # Add your custom losses here
        return custom_loss
```

## Performance Monitoring

### **TensorBoard Logging**
The system provides comprehensive logging for monitoring training:

```python
# Loss tracking
writer.add_scalar('Loss/Classification', loss_cls.item(), step)
writer.add_scalar('Loss/Router', loss_router.item(), step)
writer.add_scalar('Loss/Sparsity', loss_sparsity.item(), step)
writer.add_scalar('Loss/Total', total_loss.item(), step)

# Attention analysis
writer.add_histogram('Attention/layer_0', attention['layer_0'], step)
writer.add_scalar('Attention/layer_0_entropy', entropy, step)

# Task performance
writer.add_scalar('Accuracy/Average', avg_acc, task_id)
writer.add_scalar('Accuracy/Task_1', task_1_acc, task_id)
```

### **Key Metrics to Monitor**
1. **Router Stability**: Attention entropy, drift penalty
2. **Module Utilization**: Per-module attention averages
3. **Task Performance**: Accuracy on each task
4. **Collapse Detection**: Max attention weights, collapse steps

## Best Practices

### **Hyperparameter Tuning**
1. **Start Conservative**: Use default coefficients first
2. **Monitor Attention**: Watch for collapse or uniform attention
3. **Adjust Gradually**: Small changes to coefficients
4. **Task-Specific**: Different tasks may need different settings

### **Common Issues and Solutions**

#### **Router Collapse**
- **Symptom**: One module gets >95% attention
- **Solution**: Increase `entropy_coef`, decrease `lr_router`
- **Prevention**: Use curriculum learning, monitor attention

#### **Uniform Attention**
- **Symptom**: All modules get equal attention (~25% for 4 modules)
- **Solution**: Decrease `entropy_coef`, increase `sparsity_coef`
- **Prevention**: Use sparsity regularization

#### **Unstable Training**
- **Symptom**: High variance in losses, attention patterns
- **Solution**: Increase `drift_coef`, decrease learning rates
- **Prevention**: Use curriculum learning, monitor drift

### **Reproducibility**
```python
# Set seeds for reproducibility
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device.type == 'cuda':
    torch.cuda.manual_seed(args.seed)
```

## Files

- `train_splitmnist_balanced.py`: Main balanced training implementation
- `example_balanced_training.py`: Comprehensive usage examples
- `Documentation/BALANCED_TRAINING_README.md`: This documentation

## Testing

Run the example script to verify everything works:
```bash
python example_balanced_training.py
```

This will run all demonstrations and show:
- ✅ Basic balanced training
- ✅ Loss component analysis
- ✅ Router collapse detection
- ✅ Curriculum learning
- ✅ Two-optimizer system

## Success Criteria

The balanced training system is designed to achieve:
- **>97% accuracy** on SplitMNIST with 5 tasks
- **Stable attention patterns** without collapse
- **Efficient module utilization** (not all modules active)
- **Reproducible results** with proper seeding

## Next Steps

1. **Experiment** with different coefficient combinations
2. **Extend** to more complex datasets (CIFAR-10/100)
3. **Analyze** attention patterns across tasks
4. **Compare** with baseline methods (SI/EWC/ER)
5. **Optimize** hyperparameters for specific scenarios

The balanced training system provides a robust foundation for modular continual learning with stable attention routing!
