# SplitMNIST Experiment Harness

A reproducible experiment framework for testing continual learning with dynamic routing and module expansion on SplitMNIST.

## Overview

This harness implements a **2×2 ablation study**:
- **Router**: `{none, attn}`
- **Module Expansion**: `{off, on}`

Each configuration is run with multiple seeds, and results are aggregated into a single CSV file.

## Quick Start

### Basic Usage

Run all experiments with default settings:

```bash
python -m experiments.run_experiments
```

This will:
1. Run all 4 configurations (2 router types × 2 expansion settings)
2. Use 3 random seeds (0, 1, 2)
3. Train for 5 epochs per task
4. Output results to `results_for_fyp/splitmnist_<timestamp>/results.csv`

### Custom Configuration

```bash
python -m experiments.run_experiments \
  --seeds 0 1 2 3 4 \
  --epochs-per-task 10 \
  --batch-size 128 \
  --lr 1e-3 \
  --device cuda
```

### Single Configuration (No Grid)

To run just one configuration instead of the full grid:

```bash
python -m experiments.run_experiments \
  --run-full-grid False \
  --router attn \
  --module-expansion on \
  --seeds 0
```

## Architecture

### Components

1. **`run_experiments.py`**: Main orchestrator
   - Parses CLI arguments
   - Builds configuration grid
   - Runs experiments
   - Aggregates results into CSV

2. **`train_splitmnist.py`**: Task-incremental training loop
   - Implements SplitMNIST protocol (5 tasks: {0,1}, {2,3}, {4,5}, {6,7}, {8,9})
   - Multi-head architecture (task-incremental)
   - Tracks metrics per task

3. **`metrics.py`**: Metric computation
   - Per-task accuracy
   - Average accuracy
   - Forgetting (Chaudhry et al.)
   - Parameter counts

4. **`expansion.py`**: Module expansion logic
   - Tracks routing confidence
   - Triggers expansion when confidence < threshold
   - Cooldown mechanism

5. **`utils.py`**: Reproducibility utilities
   - Global seed setting
   - Environment metadata collection
   - Device auto-detection

### Model Architecture

**SimpleModularMLP**:
- Input: 784 (28×28 MNIST)
- Hidden: 400 units
- Task heads: 5 × 2-way classifiers (multi-head)

**Router Options**:
- `none`: Fixed deterministic routing (selects first module)
- `attn`: Attention-based routing with learned weights

## Protocol: Task-Incremental (Multi-Head)

- **5 tasks**: Binary classification on digit pairs
  - Task 0: {0 vs 1}
  - Task 1: {2 vs 3}
  - Task 2: {4 vs 5}
  - Task 3: {6 vs 7}
  - Task 4: {8 vs 9}

- **Multi-head**: Separate classifier head per task
- **Evaluation**: At test time, use the correct head for each task

## Output Structure

After running experiments, you'll find:

```
results_for_fyp/
  splitmnist_20250422_143052/
    results.csv              # Main results file
    config.json              # CLI arguments
    env.json                 # Environment metadata
    logs/
      run.log                # Execution log
    per_seed_json/
      router-none_exp-off_seed0.json
      router-none_exp-off_seed1.json
      router-attn_exp-on_seed0.json
      ...
```

## CSV Schema

Columns in `results.csv`:

**Environment**:
- `timestamp`, `commit_sha`, `device`

**Configuration**:
- `seed`, `router`, `module_expansion`, `expansion_threshold`, `expansion_cooldown`, `max_modules_per_layer`

**Training Setup**:
- `num_tasks`, `epochs_per_task`, `batch_size`, `lr`, `optimizer`

**Results**:
- `task0_acc`, `task1_acc`, `task2_acc`, `task3_acc`, `task4_acc`
- `final_avg_acc`, `final_forgetting`, `final_params`, `peak_params`
- `train_time_s`, `peak_vram_mb`

**Literature Comparison**:
- `lit_paper_key`, `lit_reported_avg_acc`, `protocol_match`

**Notes**:
- `notes` (e.g., "sanity_low_acc_task0")

## Metrics

### Average Accuracy
Final average across all 5 tasks after training.

### Forgetting
Following Chaudhry et al.:
```
forgetting_t = max_accuracy_t - final_accuracy_t
avg_forgetting = mean(forgetting for tasks 0-3)
```
(Task 4 excluded as it was just learned)

### Sanity Check
After Task 0, accuracy should be ≥ 0.90 for baseline (router=none, expansion=off).
If not, a warning is logged in `notes`.

## Command-Line Arguments

### Core Ablation
- `--router {none,attn}`: Router type (default: `attn`)
- `--module-expansion {on,off}`: Enable expansion (default: `off`)
- `--expansion-threshold FLOAT`: Confidence threshold (default: `0.2`)
- `--expansion-cooldown INT`: Tasks between expansions (default: `1`)
- `--max-modules-per-layer INT`: Max modules (default: `1`)

### Training
- `--seeds INT [INT ...]`: Random seeds (default: `0 1 2`)
- `--epochs-per-task INT`: Epochs per task (default: `5`)
- `--batch-size INT`: Batch size (default: `64`)
- `--lr FLOAT`: Learning rate (default: `3e-4`)
- `--optimizer {adam,sgd}`: Optimizer (default: `adam`)
- `--device {auto,cuda,cpu}`: Device (default: `auto`)

### Protocol
- `--num-tasks INT`: Number of tasks (default: `5`)
- `--task-incremental`: Use multi-head (default: `True`)

### I/O
- `--outdir PATH`: Output directory (default: `results_for_fyp`)
- `--lit-file PATH`: Literature baselines YAML (default: `literature/baselines_splitmnist.yaml`)
- `--data-dir PATH`: Data directory (default: `./store/datasets`)

### Execution
- `--run-full-grid`: Run full 2×2 grid (default: `True`)

## Reproducibility

### Determinism
All experiments are fully deterministic given the same seed:
- `random`, `numpy`, `torch` seeds set
- cuDNN deterministic mode enabled
- No stochastic data augmentation

### Environment Tracking
Each run records:
- Git commit SHA (if in a git repo)
- Git dirty flag (uncommitted changes)
- PyTorch version
- CUDA version
- Device name
- Timestamp

## Literature Baselines

Edit `literature/baselines_splitmnist.yaml` to add published results for comparison.

Example entry:
```yaml
- paper_key: "Zenke2017_SI"
  protocol: "task-incremental-multihead"
  heads: "multi"
  epochs_per_task: 5
  optimizer: "adam"
  reported_avg_acc: 0.965
  notes: "Synaptic Intelligence regularization."
```

The `protocol_match` column in the CSV indicates if the protocol matches literature.

## Troubleshooting

### CUDA Out of Memory
Reduce batch size:
```bash
python -m experiments.run_experiments --batch-size 32
```

### Slow Execution
Run fewer seeds or use GPU:
```bash
python -m experiments.run_experiments --seeds 0 1 --device cuda
```

### Missing Dependencies
Install requirements:
```bash
pip install torch torchvision numpy pyyaml
```

## Example Output

```
======================================================================
SplitMNIST Experiment Harness - 2×2 Ablation Study
======================================================================

Configuration:
  Seeds: [0, 1, 2]
  Epochs per task: 5
  Batch size: 64
  Learning rate: 0.0003
  Optimizer: adam
  Device: auto

Output directory: results_for_fyp/splitmnist_20250422_143052

Running 4 configurations × 3 seeds = 12 total experiments

======================================================================
Configuration 1/4
  Router: none
  Module Expansion: off
======================================================================

--- Seed 0 (1/3) ---

============================================================
Training Task 0 (classes 0-1)
============================================================
  Epoch 1/5: Loss=0.3214, Acc=0.9123
  ...

After Task 0:
  Task 0 Accuracy: 0.9856
  Average Accuracy: 0.9856
  Parameters: 634,402

...

Results written to: /path/to/results_for_fyp/splitmnist_20250422_143052/results.csv

======================================================================
EXPERIMENT COMPLETE
======================================================================

Total runs: 12
Results CSV: /path/to/results.csv
Output directory: /path/to/splitmnist_20250422_143052

Aggregate Results:
----------------------------------------------------------------------

none   + expansion=off:
  Avg Acc:   0.9456
  Forgetting: 0.0234
  Params:     634,402

attn   + expansion=off:
  Avg Acc:   0.9523
  Forgetting: 0.0198
  Params:     642,850

...
```

## Citation

If you use this experiment harness, please cite:

```
@misc{splitmnist_harness_2025,
  title={Reproducible SplitMNIST Experiment Harness},
  author={Your Name},
  year={2025},
  url={https://github.com/your/repo}
}
```

