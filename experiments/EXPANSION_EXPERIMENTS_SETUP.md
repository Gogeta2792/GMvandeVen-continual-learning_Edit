# Module Expansion Experiments - Setup Complete

## Summary

The codebase has been updated to run 4 experiments testing different combinations of attention routing and module expansion on SplitMNIST.

## The 4 Experiments

When you run `python -m experiments.run_experiments` (with default settings), it will run:

1. **Router=none, Expansion=off** (Baseline)
   - Single module, no routing
   - Fixed architecture
   - ~478k parameters

2. **Router=none, Expansion=on**
   - Starts with 1 module
   - Can expand up to 8 modules when confidence is low
   - Note: Uses routing internally for expansion decisions

3. **Router=attn, Expansion=off**
   - Starts with 2 modules
   - Attention-based routing
   - Fixed architecture
   - ~487k parameters

4. **Router=attn, Expansion=on** (Full system)
   - Starts with 2 modules
   - Attention-based routing
   - Can expand up to 8 modules
   - Dynamic architecture

Each configuration is run with 3 random seeds (0, 1, 2) by default, for a total of **12 experiments**.

## Quick Start

### Run all 4 experiments (default behavior):
```bash
python -m experiments.run_experiments
```

### Quick test (1 config, 1 seed, 2 epochs):
```bash
python -m experiments.run_experiments --single-config --router none --module-expansion off --seeds 0 --epochs-per-task 2
```

### Full production run (5 seeds, 20 epochs):
```bash
python -m experiments.run_experiments --seeds 0 1 2 3 4 --epochs-per-task 20
```

## Key Parameters Changed

| Parameter | Old Default | New Default | Reason |
|-----------|-------------|-------------|--------|
| `--max-modules-per-layer` | 1 | 8 | Allow expansion to actually add modules |
| `--expansion-threshold` | 0.2 | 0.4 | More conservative expansion trigger |
| `--expansion-cooldown` | 1 | 0 | No forced waiting between expansions |

## Output

Results are saved to `results_for_fyp/splitmnist_<timestamp>/`:
- `results.csv` - Main results file with all experiments
- `config.json` - Configuration used
- `env.json` - Environment metadata (git commit, device, etc.)
- `per_seed_json/` - Individual experiment results

## Expected Results

After training (5 epochs per task):

| Configuration | Avg Acc | Forgetting | Params |
|--------------|---------|------------|--------|
| none + off   | 0.94-0.96 | 0.02-0.04 | ~478k |
| none + on    | ? | ? | ~478k-650k* |
| attn + off   | 0.95-0.97 | 0.01-0.03 | ~487k |
| attn + on    | ? | ? | ~487k-750k* |

*Final parameter count depends on how many modules were added during expansion

## Verification

To verify the setup works, run the component tests:
```bash
python experiments/test_components.py
```

## Implementation Notes

### Expansion Mechanism

The module expansion system works as follows:

1. **Confidence Tracking**: During training, the router's attention weights are monitored
   - High confidence (max weight > threshold): Model is sure about module selection
   - Low confidence (max weight < threshold): Model is uncertain, suggests need for new module

2. **Expansion Trigger**: After each task, if average confidence is below threshold and cooldown has passed:
   - A new module is added to the layer
   - New module is initialized from EMA of existing modules
   - Router is expanded to output M+1 logits
   - Optimizer is updated with new parameters

3. **Router='none' + Expansion='on'**: 
   - This configuration uses routing internally for expansion decisions
   - The router is needed to compute confidence scores
   - See note in console output when running this configuration

### Code Changes

1. **experiments/run_experiments.py**:
   - Updated default `max_modules_per_layer` from 1 to 8
   - Updated default `expansion_threshold` from 0.2 to 0.4
   - Updated default `expansion_cooldown` from 1 to 0
   - Added clear printing of the 4 experiments being run

2. **experiments/train_splitmnist.py**:
   - Added warning message for router='none' + expansion='on' case
   - Expansion is already implemented and working in lines 391-434

3. **experiments/QUICKSTART.md**:
   - Added explicit list of 4 experiments
   - Updated parameter defaults table
   - Added note about router='none' + expansion='on'

4. **models/modular_layer.py**:
   - Expansion code already fully implemented (lines 402-720)
   - No changes needed

## Next Steps

1. Run the experiments: `python -m experiments.run_experiments`
2. Check results in `results_for_fyp/splitmnist_<timestamp>/results.csv`
3. Analyze with pandas or Excel (see QUICKSTART.md for examples)
4. Increase epochs/seeds for publication-quality results

## Troubleshooting

**Issue**: Expansion never happens
- Check that `max_modules_per_layer` > initial modules
- Lower `expansion_threshold` to trigger more easily
- Check console output for expansion trigger messages

**Issue**: Too many expansions
- Increase `expansion_threshold` 
- Increase `expansion_cooldown`

**Issue**: CUDA out of memory
- Reduce batch size: `--batch-size 32`
- Reduce max modules: `--max-modules-per-layer 4`

## Support

See also:
- `experiments/QUICKSTART.md` - Detailed usage guide
- `experiments/README.md` - Architecture documentation
- `Documentation/MODULE_EXPANSION_README.md` - Expansion mechanism details

