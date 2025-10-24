# Quick Start Guide - SplitMNIST Experiments

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Run Experiments

### 1. Full 2×2 Ablation (Recommended)

Run all 4 configurations with 3 seeds each:

```bash
python -m experiments.run_experiments
```

**Expected output:**
- 12 total experiments (4 configs × 3 seeds)
- Results in `results_for_fyp/splitmnist_<timestamp>/results.csv`
- Runtime: ~10-15 minutes on CPU (5 epochs/task)

### 2. Quick Test (Single Config, 1 Seed)

Fast test to verify everything works:

```bash
python -m experiments.run_experiments 
  --single-config 
  --router none 
  --module-expansion off 
  --seeds 0 
  --epochs-per-task 2
```

**Expected output:**
- 1 experiment
- Avg Acc: ~0.97-0.99
- Runtime: ~30 seconds

### 3. Full Training (Production Quality)

Run with more epochs for publication-quality results:

```bash
python -m experiments.run_experiments 
  --seeds 0 1 2 3 4 
  --epochs-per-task 20 
  --batch-size 128 
  --lr 1e-3 
  --device cuda
```

**Expected output:**
- 20 experiments (4 configs × 5 seeds)
- Higher accuracy due to more training
- Runtime: ~2-3 hours on GPU

### 4. Custom Configuration

Test specific ablation:

```bash
python -m experiments.run_experiments \
  --single-config \
  --router attn \
  --module-expansion on \
  --expansion-threshold 0.3 \
  --max-modules-per-layer 3 \
  --seeds 0 1 2
```

## Verify Installation

Run component tests:

```bash
python experiments/test_components.py
```

**Expected output:**
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

## Understanding the Output

### Directory Structure

After running experiments:

```
results_for_fyp/
  splitmnist_20251022_143052/
    results.csv              # ← Main results file (open with Excel/pandas)
    config.json              # CLI arguments used
    env.json                 # Git commit, device info, etc.
    logs/
      run.log                # Execution log
    per_seed_json/
      router-none_exp-off_seed0.json
      router-attn_exp-on_seed1.json
      ...
```

### CSV Columns

**Key columns to look at:**
- `router`: {none, attn}
- `module_expansion`: {off, on}
- `seed`: Random seed (or "agg_mean", "agg_std")
- `final_avg_acc`: Average accuracy across all 5 tasks
- `final_forgetting`: How much the model forgot previous tasks
- `final_params`: Number of trainable parameters

**Aggregate rows:**
- `seed=agg_mean`: Mean across all seeds for this config
- `seed=agg_std`: Standard deviation across seeds

### Expected Results

For baseline (router=none, expansion=off, 5 epochs):
- **Avg Acc**: 0.94-0.96
- **Forgetting**: 0.02-0.04
- **Params**: ~478k

With attention router (router=attn, expansion=off):
- **Avg Acc**: 0.95-0.97 (slightly better)
- **Forgetting**: 0.01-0.03 (less forgetting)
- **Params**: ~487k (slightly more due to router)

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution**: Reduce batch size
```bash
python -m experiments.run_experiments --batch-size 32
```

### Issue: "No module named 'torchvision'"

**Solution**: Install missing dependency
```bash
pip install torchvision pyyaml
```

### Issue: Experiments are slow

**Solutions:**
1. Use GPU: `--device cuda`
2. Reduce epochs: `--epochs-per-task 3`
3. Run single config: `--single-config --router none --module-expansion off`

### Issue: Results are not reproducible

**Check:**
1. Are you using the same seed? `--seeds 0`
2. Is the code unchanged? (Check `git_dirty` in `env.json`)
3. Same PyTorch version? (Check `env.json`)

## Analyzing Results

### With Python/pandas

```python
import pandas as pd

# Load results
df = pd.read_csv('results_for_fyp/splitmnist_<timestamp>/results.csv')

# Filter to aggregate rows only
agg = df[df['seed'] == 'agg_mean']

# Compare configurations
print(agg[['router', 'module_expansion', 'final_avg_acc', 'final_forgetting']])

# Plot results
import matplotlib.pyplot as plt
agg['config'] = agg['router'] + ' + ' + agg['module_expansion']
agg.plot(x='config', y='final_avg_acc', kind='bar')
plt.ylabel('Average Accuracy')
plt.title('SplitMNIST Results')
plt.show()
```

### With Excel

1. Open `results.csv` in Excel
2. Filter `seed` column to "agg_mean"
3. Create pivot table or chart comparing configurations

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

### Execution Modes

| Flag | Description |
|------|-------------|
| `--run-full-grid` | Run all 4 configurations (default) |
| `--single-config` | Run only specified configuration |

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

