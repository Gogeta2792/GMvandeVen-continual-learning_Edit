# Setup and Installation

This guide covers installation, prerequisites, and how to run the Modular Continual Learning system.

## Prerequisites

### System Requirements
- **Python**: 3.10.4 (tested version)
- **PyTorch**: 1.11.0
- **Torchvision**: 0.12.0
- **Operating System**: Linux/macOS/Windows (tested on Fedora)

### Hardware Requirements
- **CPU**: Standard desktop computer
- **GPU**: Optional but recommended for faster training
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Storage**: 2GB free space for datasets and models

## Installation

### 1. Install Dependencies

Install all required Python packages:

```bash
pip install -r requirements.txt
```

### 2. Make Scripts Executable (Linux/macOS)

```bash
chmod +x main*.py compare*.py all_results.sh
```

### 3. Verify Installation

Run a quick test to verify everything works:

```bash
python -c "import torch; import torchvision; print('Installation successful!')"
```

## Quick Start

### 1. Sanity Check

Run a small single experiment on Split MNIST with Synaptic Intelligence (SI):

```bash
./main.py --experiment=splitMNIST --scenario=task --si
```

**Expected runtime**: ~6 minutes on CPU (~3 minutes on GPU)

The script prints data/model details, training progress, and outputs.

### 2. Compare Methods

Run multiple methods on Split MNIST (task-incremental) and summarize results:

```bash
./compare.py --experiment=splitMNIST --scenario=task
```

**Expected runtime**: ~100 minutes on CPU (~45 minutes on GPU)

Produces logs and a summary PDF.

## Running Experiments

### Academic Continual Learning Setting

Use `main.py` with common options:

#### Basic Parameters
- `--experiment`: `splitMNIST|permMNIST|CIFAR10|CIFAR100`
- `--contexts`: Number of contexts
- `--scenario`: `task|domain|class`

#### Method Examples

**Baselines**:
```bash
./main.py                    # None (lower target)
./main.py --joint            # Joint (upper target)
```

**Regularization/Replay Methods**:
```bash
./main.py --ewc              # EWC (see ICLRblogpost/README.md first)
./main.py --si               # Synaptic Intelligence
./main.py --lwf              # Learning without Forgetting
./main.py --fromp            # FROMP
./main.py --replay=buffer    # ER
./main.py --agem             # A-GEM
./main.py --replay=generative# DGR
./main.py --icarl            # iCaRL
```

**Other Approaches**:
```bash
./main.py --separate-networks
./main.py --xdg              # Context-dependent gating (XdG)
```

**Discover Options**:
```bash
./main.py -h
```

### Task-Free Continual Learning

Run more flexible streams without explicit boundaries using `main_task_free.py`:

#### Basic Parameters
- `--experiment`: `splitMNIST|permMNIST|CIFAR10|CIFAR100`
- `--contexts`: Number of contexts
- `--stream`: `fuzzy-boundaries|academic-setting|random`
- `--scenario`: `task|domain|class`

#### Example
```bash
./main_task_free.py --experiment=splitMNIST --stream=fuzzy-boundaries --scenario=task --si --update-every=500
```

**Check all options**:
```bash
./main_task_free.py -h
```

## Modular Training

### Balanced Training

Run balanced training with modular layers:

```bash
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

### Module Expansion

Run with dynamic module expansion:

```bash
python train_splitmnist_modular.py \
    --experiment splitMNIST \
    --scenario task \
    --contexts 5 \
    --num_modules 2 \
    --max_modules 6 \
    --enable_expansion \
    --confidence_threshold 0.4 \
    --cooldown_steps 500
```

## Experiment Harness

### Full 2×2 Ablation Study

Run all experiments with default settings:

```bash
python -m experiments.run_experiments
```

This runs:
- 4 configurations (2 router types × 2 expansion settings)
- 3 seeds (0, 1, 2)
- 5 epochs per task
- Total: 12 experiments

### Single Configuration

```bash
python -m experiments.run_experiments \
    --single-config \
    --router attn \
    --module-expansion on \
    --seeds 0 1 2 \
    --epochs-per-task 10
```

### Custom Settings

```bash
python -m experiments.run_experiments \
    --seeds 0 1 2 3 4 \
    --epochs-per-task 20 \
    --batch-size 128 \
    --lr 1e-3 \
    --device cuda
```

### Quick Test

Fast test to verify everything works:

```bash
python -m experiments.run_experiments \
    --single-config \
    --router none \
    --module-expansion off \
    --seeds 0 \
    --epochs-per-task 2
```

**Expected output**:
- 1 experiment
- Avg Acc: ~0.97-0.99
- Runtime: ~30 seconds

## Visualization

### On-the-fly Plots with Visdom

Install and start visdom server:

```bash
pip install visdom
python -m visdom.server
```

Open `http://localhost:8097` in your browser and add `--visdom` to your run command:

```bash
./main.py --experiment=splitMNIST --scenario=task --si --visdom
```

## Data Storage

### Default Locations
- **Datasets**: `store/datasets/` (MNIST/CIFAR downloaded automatically)
- **Models**: `store/models/` (saved model checkpoints)
- **Results**: `results_for_fyp/` (experiment outputs)

### First Run
The code will download MNIST/CIFAR as needed. Ensure you have internet access for first runs.

## Troubleshooting

### Common Issues

#### CUDA Out of Memory
**Solution**: Reduce batch size
```bash
python -m experiments.run_experiments --batch-size 32
```

#### Missing Dependencies
**Solution**: Install missing packages
```bash
pip install torch torchvision numpy pyyaml visdom
```

#### Slow Execution
**Solutions**:
1. Use GPU: `--device cuda`
2. Reduce epochs: `--epochs-per-task 3`
3. Run single config: `--single-config --router none --module-expansion off`

#### Results Not Reproducible
**Check**:
1. Same seed: `--seeds 0`
2. Unchanged code: Check `git_dirty` in `env.json`
3. Same PyTorch version: Check `env.json`

#### "No module named 'torchvision'"
**Solution**: Install missing dependency
```bash
pip install torchvision pyyaml
```

### Verification

Run component tests to verify setup:

```bash
python experiments/test_components.py
```

**Expected output**:
```
======================================================================
Component Validation Tests
======================================================================
✓ PASS: Imports
✓ PASS: Utils
✓ PASS: Metrics
✓ PASS: Expansion
✓ PASS: Model Creation
✓ PASS: Data Loading

Total: 6/6 tests passed
```

## Advanced Usage

### Reproducing Paper Results

See `all_results.sh` for step-by-step instructions to recreate tables and figures from the article "Three types of incremental learning". Running as-is can be time-consuming; consider parallelizing experiments.

### Custom Experiments

#### Academic Setting
Custom individual experiments in the academic continual learning setting can be run with `main.py`. The code supports combinations of several methods and allows creating custom approaches by mixing components.

#### Task-Free Setting
Custom individual experiments in a more flexible, "task-free" continual learning setting can be run with `main_task_free.py`. Some methods have been slightly modified to make them suitable for the absence of known context boundaries.

### Integration with Existing Pipeline

The modular system works with existing `--experiment=splitMNIST --scenario=task` pipeline:

```bash
python train_splitmnist_balanced.py \
    --experiment splitMNIST \
    --scenario task \
    --contexts 5 \
    --use_router \
    --gpu
```

## Command-Line Reference

### Core Parameters

| Parameter | Values | Default | Description |
|-----------|--------|---------|-------------|
| `--router` | none, attn | attn | Router type |
| `--module-expansion` | on, off | off | Enable expansion |
| `--seeds` | int(s) | 0 1 2 | Random seeds |
| `--epochs-per-task` | int | 5 | Training epochs per task |
| `--batch-size` | int | 64 | Batch size |
| `--lr` | float | 3e-4 | Learning rate |
| `--device` | auto, cuda, cpu | auto | Device |

### Expansion Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--expansion-threshold` | float | 0.2 | Confidence threshold |
| `--expansion-cooldown` | int | 1 | Tasks between expansions |
| `--max-modules-per-layer` | int | 1 | Maximum modules |

### Balanced Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--lr_modules` | float | 0.001 | Learning rate for modules |
| `--lr_router` | float | 0.0001 | Learning rate for router |
| `--entropy_coef` | float | 0.01 | Entropy regularization coefficient |
| `--drift_coef` | float | 0.1 | Drift regularization coefficient |
| `--sparsity_coef` | float | 0.001 | Sparsity regularization coefficient |
| `--freeze_router_steps` | int | 500 | Steps to freeze router initially |
| `--collapse_threshold` | float | 0.95 | Router collapse threshold |
| `--max_collapse_steps` | int | 100 | Max steps before stopping on collapse |

## Next Steps

1. **Run baseline**: Get familiar with the output
2. **Check reproducibility**: Run same config twice, compare results
3. **Full ablation**: Run with more seeds and epochs
4. **Add literature**: Fill in `literature/baselines_splitmnist.yaml`
5. **Extend**: Add more algorithms (SI, EWC, etc.)

## Support

For issues or questions:
1. Check `experiments/README.md` for detailed documentation
2. Run `python experiments/test_components.py` to verify setup
3. Check the logs in `results_for_fyp/*/logs/run.log`
4. Review the comprehensive documentation in the `Documentation/` folder
